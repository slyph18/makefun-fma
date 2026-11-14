"""free-model-aggregator-01 entry: OpenAI/Anthropic-compatible proxy over 10 simulated free providers."""
import json, os, sys, time, threading, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENV = os.environ.get("MAKEFUN_ENV_URL") or sys.exit("MAKEFUN_ENV_URL is not set")
SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")

def env_get(path):
    with urllib.request.urlopen(ENV + path, timeout=30) as r:
        return json.load(r)

def provider_call(path, body, wire, auth, key, stream):
    """POST to a provider in its own wire. Returns (status, bytes, headers)."""
    url = ENV + path
    data = json.dumps(body).encode()
    headers = {"content-type": "application/json"}
    if auth == "bearer":
        headers["Authorization"] = "Bearer " + key
    elif auth == "x-api-key":
        headers["x-api-key"] = key
    elif auth == "x-goog-api-key":
        headers["x-goog-api-key"] = key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)

# ---------------- wire conversion ----------------
STOP_MAP_ANTH = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}
STOP_MAP_OAI = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal"}

def oai_to_anthropic(body):
    """OpenAI chat.completions request -> Anthropic messages request."""
    msgs = []
    system = ""
    for m in body.get("messages", []):
        if m.get("role") == "system":
            system += m.get("content", "")
        else:
            msgs.append({"role": m.get("role"), "content": m.get("content")})
    out = {"model": body.get("model", ""), "messages": msgs, "max_tokens": body.get("max_tokens", 4096)}
    if system:
        out["system"] = system
    if body.get("tools"):
        out["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                         "input_schema": t["function"].get("parameters", {})} for t in body["tools"]]
    if body.get("tool_choice"):
        tc = body["tool_choice"]
        if isinstance(tc, str) and tc == "auto":
            out["tool_choice"] = {"type": "auto"}
        elif isinstance(tc, dict) and tc.get("type") == "function":
            out["tool_choice"] = {"type": "tool", "name": tc["function"]["name"]}
    if body.get("stop"):
        out["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    return out

def anthropic_to_oai(resp):
    """Anthropic messages response -> OpenAI chat.completions response."""
    text_parts = []
    tool_calls = []
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({"id": block.get("id", ""), "type": "function",
                               "function": {"name": block["name"], "arguments": json.dumps(block.get("input", {}))}})
    stop = resp.get("stop_reason", "end_turn")
    msg = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    u = resp.get("usage", {})
    return {"id": resp.get("id", ""), "object": "chat.completion", "created": 0,
            "model": resp.get("model", ""), "choices": [{"index": 0, "message": msg, "finish_reason": STOP_MAP_ANTH.get(stop, stop)}],
            "usage": {"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0), "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0)}}

def oai_to_gemini(body):
    """OpenAI chat.completions request -> Gemini generateContent request."""
    contents = []
    system = None
    for m in body.get("messages", []):
        role = "user" if m.get("role") in ("user", "system") else "model"
        c = m.get("content")
        if m.get("role") == "system":
            system = c
            continue
        if isinstance(c, str):
            contents.append({"role": role, "parts": [{"text": c}]})
        elif isinstance(c, list):
            parts = []
            for seg in c:
                if seg.get("type") == "text":
                    parts.append({"text": seg.get("text", "")})
                elif seg.get("type") == "image_url":
                    url = seg.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        head, b64 = url.split(",", 1)
                        mime = head.split(":")[1].split(";")[0]
                        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            contents.append({"role": role, "parts": parts})
    out = {"contents": contents}
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = []
    fns = []
    for t in body.get("tools", []) or []:
        f = t.get("function", {})
        fns.append({"name": f.get("name"), "description": f.get("description", ""), "parameters": f.get("parameters", {})})
    if fns:
        tools.append({"functionDeclarations": fns})
    if tools:
        out["tools"] = tools
    gen = {}
    if body.get("temperature") is not None:
        gen["temperature"] = body["temperature"]
    if body.get("max_tokens"):
        gen["maxOutputTokens"] = body["max_tokens"]
    if body.get("stop"):
        gen["stopSequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    if body.get("response_format") and body["response_format"].get("type") == "json_object":
        gen["responseMimeType"] = "application/json"
    if gen:
        out["generationConfig"] = gen
    return out

def gemini_to_oai(resp, model):
    cand = (resp.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    tool_calls = []
    for p in parts:
        for fc in p.get("functionCall", []) if isinstance(p.get("functionCall"), list) else [p.get("functionCall")] if p.get("functionCall") else []:
            tool_calls.append({"id": "call_" + fc.get("name", "x"), "type": "function",
                               "function": {"name": fc.get("name"), "arguments": json.dumps(fc.get("args", {}))}})
    finish = cand.get("finishReason", "STOP")
    fmap = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter"}
    msg = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    u = resp.get("usageMetadata", {})
    return {"id": "gemini", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": fmap.get(finish, finish.lower())}],
            "usage": {"prompt_tokens": u.get("promptTokenCount", 0), "completion_tokens": u.get("candidatesTokenCount", 0), "total_tokens": u.get("totalTokenCount", 0)}}

# ---------------- catalog & routing ----------------
CAPS = {}   # (provider, model) -> {free, context_length, tools, vision, json, rating}
PROV = {}   # id -> {wire, auth, key, path, key_env}
CATALOG = None

def build_catalog(inst):
    global CATALOG
    CATALOG = inst
    plist = inst["providers"]["providers"] if isinstance(inst["providers"].get("providers"), list) else list(inst["providers"]["providers"].values())
    for p in plist:
        PROV[p["id"]] = p
        for m in p.get("models", []):
            CAPS[(p["id"], m["id"])] = {"free": m.get("free", False), "context_length": m.get("context_length", 0),
                                        "tools": m.get("tools", False), "vision": m.get("vision", False),
                                        "json": m.get("json", False), "rating": m.get("rating_milli", 0),
                                        "wire": p["wire"], "auth": p["auth"], "key": p.get("key", ""), "path": p["path"]}
    return catalog_doc()

def catalog_doc():
    provs = []
    for pid, p in PROV.items():
        free = sorted(mid for (pi, mid) in CAPS if pi == pid and CAPS[(pi, mid)]["free"])
        paid = sorted(mid for (pi, mid) in CAPS if pi == pid and not CAPS[(pi, mid)]["free"])
        provs.append({"id": pid, "base_url": p["path"], "key_env": p.get("key_env", ""), "wire": p["wire"],
                      "free_models": free, "paid_models": paid})
    return {"schema": "free-model-aggregator-01/catalog/v1", "providers": provs}

def requires_of(body):
    """Infer what the request requires from its shape."""
    reqs = {"tools": False, "vision": False, "json": bool(body.get("response_format")), "min_context": 0}
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            for seg in c:
                if seg.get("type") == "image_url":
                    reqs["vision"] = True
    if body.get("tools"):
        reqs["tools"] = True
    return reqs

USED = {}
LOCK = threading.Lock()
INFLIGHT = set()

def pick(reqs, exclude):
    best = None
    for (pid, mid), c in sorted(CAPS.items(), key=lambda kv: -kv[1]["rating"]):
        if not c["free"]:
            continue
        if (pid, mid) in exclude:
            continue
        if reqs.get("tools") and not c["tools"]:
            continue
        if reqs.get("vision") and not c["vision"]:
            continue
        if reqs.get("json") and not c["json"]:
            continue
        if reqs.get("min_context") and c["context_length"] < reqs["min_context"]:
            continue
        if USED.get((pid, mid), 0) >= BUDGET_CAP:
            continue
        best = (pid, mid, c)
        break
    return best

BUDGET_CAP = 10 ** 9

# ---------------- HTTP server ----------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            auth = self.headers.get("Authorization", "")
            free = sorted({mid for (pi, mid), c in CAPS.items() if c["free"]})
            if self.headers.get("x-api-key"):
                out = {"data": [{"type": "model", "id": mid, "display_name": mid} for mid in free]}
            else:
                out = {"object": "list", "data": [{"id": mid, "object": "model", "created": 0, "owned_by": "agg"} for mid in free]}
            self._send(200, out)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        ln = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(ln)
        try:
            body = json.loads(raw)
        except Exception:
            return self._send(400, {"error": "bad json"})
        is_anthropic = self.headers.get("x-api-key") is not None
        stream = bool(body.get("stream"))
        reqs = requires_of(body)
        exclude = set()
        for attempt in range(len(CAPS)):
            pickd = pick(reqs, exclude)
            if not pickd:
                break
            pid, mid, cap = pickd
            try:
                status, resp_bytes, hdrs = dispatch(body, pid, mid, cap, is_anthropic, stream, raw)
            except Exception:
                exclude.add((pid, mid))
                continue
            if status == 200:
                with LOCK:
                    USED[(pid, mid)] = USED.get((pid, mid), 0) + 1
                ctype = hdrs.get("content-type", "application/json")
                self._send(200, resp_bytes, ctype)
                return
            with LOCK:
                FAILS.append((pid, mid))
            exclude.add((pid, mid))
        self._send(502, {"error": {"message": "no free provider could serve this request"}})

FAILS = []

def dispatch(body, pid, mid, cap, is_anthropic, stream, raw):
    p = PROV[pid]
    wire = cap["wire"]
    if wire == "openai":
        out_body = dict(body)
        out_body["model"] = mid
        path = p["path"] + "/v1/chat/completions"
    elif wire == "anthropic":
        out_body = oai_to_anthropic(body) if not is_anthropic else dict(body)
        out_body["model"] = mid
        path = p["path"] + "/v1/messages"
    else:  # gemini
        out_body = oai_to_gemini(body)
        path = f"{p['path']}/v1beta/models/{mid}:generateContent"
    status, resp_bytes, hdrs = provider_call(path, out_body, wire, cap["auth"], cap["key"], stream)
    if status != 200:
        return status, resp_bytes, hdrs
    # normalize answer to the client's wire
    try:
        rj = json.loads(resp_bytes)
    except Exception:
        return status, resp_bytes, hdrs
    if wire == "openai":
        if is_anthropic:
            rj = oai_resp_to_anthropic(rj, mid)
            return 200, json.dumps(rj).encode(), {"content-type": "application/json"}
        return 200, resp_bytes, {"content-type": "application/json"}
    if wire == "anthropic":
        if is_anthropic:
            return 200, resp_bytes, {"content-type": "application/json"}
        return 200, json.dumps(anthropic_to_oai(rj)).encode(), {"content-type": "application/json"}
    # gemini -> whichever client wire
    oai = gemini_to_oai(rj, mid)
    if is_anthropic:
        return 200, json.dumps(oai_resp_to_anthropic(oai, mid)).encode(), {"content-type": "application/json"}
    return 200, json.dumps(oai).encode(), {"content-type": "application/json"}

def oai_resp_to_anthropic(rj, model):
    ch = (rj.get("choices") or [{}])[0]
    msg = ch.get("message", {})
    content = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls", []) or []:
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        content.append({"type": "tool_use", "id": tc.get("id", "call"), "name": tc["function"]["name"], "input": args})
    fmap = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal"}
    u = rj.get("usage", {})
    return {"id": rj.get("id", "msg"), "type": "message", "role": "assistant", "model": model,
            "content": content or [{"type": "text", "text": ""}],
            "stop_reason": fmap.get(ch.get("finish_reason", "stop"), "end_turn"),
            "usage": {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}}

def main():
    if "--list" in sys.argv:
        inst = env_get("/instance")
        build_catalog(inst)
        print(json.dumps(catalog_doc()))
        return
    inst = env_get("/instance")
    catalog = build_catalog(inst)
    port = int(os.environ.get("PORT", "0"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    port = srv.server_address[1]
    body = json.dumps({"protocol_version": 1, "port": port, "catalog": catalog}).encode()
    req = urllib.request.Request(ENV + "/register", data=body, headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        reg = json.load(r)
    print(f"registered: {reg} port={port}", file=sys.stderr)
    import threading as th
    t = th.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    while True:
        with urllib.request.urlopen(ENV + "/work", timeout=30) as r:
            w = json.load(r)
        if w.get("done"):
            break
        time.sleep(1.0)
    srv.shutdown()
    srv.server_close()
    print("done", file=sys.stderr)

if __name__ == "__main__":
    main()
