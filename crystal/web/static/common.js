const C = {
  esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  },
  pct(p, d = 1) { return p == null ? "–" : (p * 100).toFixed(d) + "%"; },
  cls(label) { return label === 1 ? "fake" : "true"; },
  name(label) { return label === 1 ? "FAKE" : "TRUE"; },
  chip(label) { return `<span class="chip ${C.cls(label)}">${C.name(label)}</span>`; },
  q() { return new URLSearchParams(location.search); },
  async get(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
  },
  header() {
    const q = C.q().get("q") || "";
    return `<header class="header"><div class="header-in">
      <a class="logo" href="/"><svg viewBox="0 0 24 24" fill="none"><path d="M12 2 2 9l10 13 10-13-10-7z" fill="#1b8a78"/><path d="M2 9h20L12 22 2 9z" fill="#0f5e52"/><path d="M7 2 2 9h20l-5-7H7z" fill="#5fb8a6"/></svg>CRYSTAL</a>
      <form class="search" action="/" method="get"><input name="q" placeholder="Search by keyword" value="${C.esc(q)}" autocomplete="off"><button type="submit" aria-label="search"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg></button></form>
      <button class="logout" type="button">log out</button>
      <button class="switch-user" type="button">switch user</button></div></header>`;
  },
  bindUserMenu() {
    const btn = document.querySelector(".switch-user");
    if (!btn) return;
    btn.addEventListener("click", () => C.openUserMenu());
  },
  async openUserMenu() {
    document.querySelector(".um-back")?.remove();
    const d = await C.get("/api/demo-users");
    const cur = C.q().get("user_id") || d.default_user;
    const back = document.createElement("div");
    back.className = "um-back";
    const rows = d.users.map(u => `<div class="um-row ${u.user_id === cur ? "on" : ""}">
        <button class="um-pick" data-id="${C.esc(u.user_id)}"><span class="um-name">${C.esc(u.name)}</span></button>
        <button class="um-reset" data-id="${C.esc(u.user_id)}">reset</button></div>`).join("");
    back.innerHTML = `<div class="um"><div class="um-top"><h3>Select user</h3><button class="um-x" aria-label="close">×</button></div>${rows}</div>`;
    document.body.appendChild(back);
    back.addEventListener("click", e => { if (e.target === back) back.remove(); });
    back.querySelector(".um-x").onclick = () => back.remove();
    back.querySelectorAll(".um-pick").forEach(b => b.onclick = () => { location.href = `/?user_id=${encodeURIComponent(b.dataset.id)}`; });
    back.querySelectorAll(".um-reset").forEach(b => b.onclick = async () => {
      await C.post("/api/reset-user", { user_id: b.dataset.id });
      if (b.dataset.id === cur) location.reload(); else C.openUserMenu();
    });
  },
  bar(d, big = false) {
    const t = d.true || 0, f = d.fake || 0;
    const seg = (k, lab, v) => v > 0 ? `<div class="seg ${k}" style="flex:${v} 0 0"><span>${lab}</span><span>${Math.round(v * 100)}%</span></div>` : "";
    return `<div class="bar">${seg("t", "True", t)}${seg("f", "Fake", f)}</div>`;
  },
  paragraphBreaks(text) {
    if (text.includes("\n")) return [];
    const re = /[.!?]["”’']?\s+(?=["“‘]?[A-Z])/g;
    const breaks = [];
    let m, n = 0, last = 0;
    while ((m = re.exec(text))) {
      const at = m.index + m[0].length;
      n++;
      if (n >= 3 || at - last >= 450) { breaks.push(at); n = 0; last = at; }
    }
    return breaks;
  },
  highlight(text, keywords, field, label) {
    const ks = (keywords || []).filter(k => k.field === field && k.start >= 0 && k.end <= text.length && k.end > k.start).sort((a, b) => a.start - b.start);
    const breaks = field === "body" ? C.paragraphBreaks(text) : [];
    let out = "", pos = 0, i = 0;
    const plain = (from, to) => {
      let s = "", p = from;
      for (const b of breaks) if (b > from && b < to) { s += C.esc(text.slice(p, b)) + "</p><p>"; p = b; }
      return s + C.esc(text.slice(p, to)).replace(/\n+/g, "</p><p>");
    };
    for (const k of ks) {
      if (k.start < pos) continue;
      out += plain(pos, k.start);
      out += `<span class="kw ${C.cls(label)}" data-i="${i++}" data-phrase="${C.esc(k.phrase)}" data-field="${field}">${C.esc(text.slice(k.start, k.end))}</span>`;
      pos = k.end;
    }
    out += plain(pos, text.length);
    return field === "body" ? `<p>${out}</p>` : out;
  },
  bindKeywords(root, art, label) {
    const cache = {};
    root.querySelectorAll(".kw").forEach(span => {
      span.addEventListener("click", async () => {
        const phrase = span.dataset.phrase;
        const block = span.closest("p, h1");
        const open = block.nextElementSibling;
        if (open && open.classList.contains("explain") && open.dataset.phrase === phrase) {
          open.remove(); span.classList.remove("active"); return;
        }
        root.querySelectorAll(".explain").forEach(e => e.remove());
        root.querySelectorAll(".kw.active").forEach(e => e.classList.remove("active"));
        span.classList.add("active");
        const box = document.createElement("div");
        box.className = `explain ${C.cls(label)}`;
        box.dataset.phrase = phrase;
        const known = (art.rationales || {})[phrase] || cache[phrase];
        box.innerHTML = `<div class="ph">${C.esc(phrase)}</div><div class="tx ${known ? "" : "pending"}">${C.esc(known || "Analyzing context...")}</div><button class="x" aria-label="close">×</button>`;
        box.querySelector(".x").onclick = () => { box.remove(); span.classList.remove("active"); };
        block.insertAdjacentElement("afterend", box);
        if (!known) {
          try {
            const r = await C.post("/api/rationale", { article_id: art.article_id, kind: "keyword", key: phrase });
            if (r.status !== "pending") cache[phrase] = r.text;
            const tx = box.querySelector(".tx");
            tx.textContent = r.text;
            tx.classList.toggle("pending", r.status === "pending");
          } catch (e) { box.querySelector(".tx").textContent = "Explanation unavailable: " + e.message; }
        }
      });
    });
  }
};
