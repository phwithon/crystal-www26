(async function () {
  window.addEventListener("pageshow", e => { if (e.persisted) location.reload(); });  // 뒤로가기 캐시 복원 시 통계 다시 받기
  document.getElementById("hdr").innerHTML = C.header();
  C.bindUserMenu();
  const q = C.q();
  const user = q.get("user_id") || "";
  const page = parseInt(q.get("page") || "1", 10);
  const search = q.get("q");
  const list = document.getElementById("list");
  const side = document.getElementById("side");
  const settings = await C.get("/api/settings");
  document.title = "CRYSTAL";
  document.getElementById("goal").textContent = settings.goals.feed;
  document.getElementById("goal").hidden = !settings.show_level_caption;

  const uid = user || settings.default_user;
  const showLabels = settings.show_feed_labels;
  const row = a => `<a class="feed-row ${showLabels ? "" : "nolabel"}" href="/article/${encodeURIComponent(a.article_id)}?user_id=${encodeURIComponent(uid)}" data-id="${C.esc(a.article_id)}">
      <div class="lab">${C.chip(a.final_label)}<span class="prob">${C.pct(a.prob)}</span></div>
      <div class="ttl">${C.esc(a.title)}</div></a>`;

  let data;
  if (search) {
    data = await C.get(`/api/search?q=${encodeURIComponent(search)}`);
    list.innerHTML = data.items.length ? data.items.map(row).join("") : `<div class="empty">No articles found for “${C.esc(search)}”</div>`;
    const f = await C.get(`/api/feed?user_id=${encodeURIComponent(uid)}&page=1`);
    renderSide(f.stats);
  } else {
    data = await C.get(`/api/feed?user_id=${encodeURIComponent(uid)}&page=${page}`);
    const pager = [];
    const link = p => `/?user_id=${encodeURIComponent(uid)}&page=${p}`;
    pager.push(`<button ${data.page <= 1 ? "disabled" : ""} onclick="location.href='${link(data.page - 1)}'">‹</button>`);
    for (let p = 1; p <= data.pages; p++) pager.push(`<button class="${p === data.page ? "on" : ""}" onclick="location.href='${link(p)}'">${p}</button>`);
    pager.push(`<button ${data.page >= data.pages ? "disabled" : ""} onclick="location.href='${link(data.page + 1)}'">›</button>`);
    list.innerHTML = (data.items.length ? data.items.map(row).join("") : `<div class="empty">no articles available</div>`) + `<div class="pager">${pager.join("")}</div>`;
    renderSide(data.stats);
  }

  function renderSide(s) {
    side.innerHTML = `<div class="card"><h3>My Weekly Veracity Stats</h3>${C.bar(s.weekly)}<h3>My Overall Veracity Stats</h3>${C.bar(s.overall)}</div>
      <div class="card"><h3>Veracity in My Feed</h3>${C.bar(s.feed)}<div class="bar-foot"><span>TRUE ${s.feed.n_true} · FAKE ${s.feed.n_fake}</span><span>TOTAL ${s.feed.total}</span></div></div>`;
  }
})();
