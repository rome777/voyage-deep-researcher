// 바다 격자와 역사 항로를 만든다 (LLM 호출 없음).
//   node tools/build_seagrid.mjs
// - concepts/vendor/land-50m.json(Natural Earth) 을 0.25° 격자로 래스터화해 바다/육지 칸을 나눈다
// - 좁은 해협은 실제 수로를 따라 바다로 연다
// - 역사 항로는 사료에 나온 기항지·경유지만 '제어점'으로 두고, 그 사이를 바다 칸으로만 잇는다 (A*)
// - 결과: data/seagrid.json  { grid, mask(base64), routes[], corridor[] }
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RES = 0.25, LON0 = -180, LAT0 = -66, LAT1 = 76;
const W = Math.round(360 / RES), H = Math.round((LAT1 - LAT0) / RES);

/* ── 1. 육지 고리 풀기 (날짜변경선을 넘는 고리는 이어 붙이고 ±360° 사본을 둔다) ── */
const topo = JSON.parse(fs.readFileSync(path.join(ROOT, 'concepts/vendor/land-50m.json'), 'utf8'));
const [sx, sy] = topo.transform.scale, [tx, ty] = topo.transform.translate;
const arcs = topo.arcs.map(a => { let x = 0, y = 0; return a.map(([dx, dy]) => { x += dx; y += dy; return [x * sx + tx, y * sy + ty]; }); });
const arcPts = i => i >= 0 ? arcs[i] : arcs[~i].slice().reverse();
const rings = [];
for (const g of topo.objects.land.geometries){
  const polys = g.type === 'MultiPolygon' ? g.arcs : [g.arcs];
  for (const poly of polys) for (const ring of poly){
    let pts = []; for (const i of ring){ const s = arcPts(i); pts = pts.concat(pts.length ? s.slice(1) : s); }
    let shift = 0, prev = null; const un = pts.map(([lo, la]) => { if (prev !== null && lo + shift - prev > 180) shift -= 360; else if (prev !== null && lo + shift - prev < -180) shift += 360; prev = lo + shift; return [prev, la]; });
    const lons = un.map(p => p[0]); const offs = [0]; if (Math.max(...lons) > 180) offs.push(-360); if (Math.min(...lons) < -180) offs.push(360);
    for (const o of offs) rings.push(un.map(([lo, la]) => [lo + o, la]));
  }
}

/* ── 2. 주사선 래스터화: 칸 중심이 육지 안이면 육지 (짝-홀 규칙, 구멍 처리 포함) ── */
const land = new Uint8Array(W * H);
const rowEdges = Array.from({length:H}, () => []);
for (const r of rings) for (let i = 0, j = r.length - 1; i < r.length; j = i++){
  const [x1, y1] = r[j], [x2, y2] = r[i]; if (y1 === y2) continue;
  const ylo = Math.min(y1, y2), yhi = Math.max(y1, y2);
  const r0 = Math.max(0, Math.ceil((ylo - LAT0) / RES - 0.5)), r1 = Math.min(H - 1, Math.floor((yhi - LAT0) / RES - 0.5));
  for (let row = r0; row <= r1; row++) rowEdges[row].push([x1, y1, x2, y2]);
}
for (let row = 0; row < H; row++){
  const lat = LAT0 + (row + 0.5) * RES, xs = [];
  for (const [x1, y1, x2, y2] of rowEdges[row]) if ((y1 > lat) !== (y2 > lat)) xs.push(x1 + (lat - y1) * (x2 - x1) / (y2 - y1));
  xs.sort((a, b) => a - b);
  for (let k = 0; k + 1 < xs.length; k += 2){
    const c0 = Math.max(0, Math.ceil((xs[k] - LON0) / RES - 0.5)), c1 = Math.min(W - 1, Math.floor((xs[k + 1] - LON0) / RES - 0.5));
    for (let c = c0; c <= c1; c++) land[row * W + c] = 1;
  }
}

/* ── 3. 좁은 해협은 실제 수로를 따라 연다 ── */
const CHANNELS = {
  '지브롤터 해협': [[-6.2,35.95],[-5.6,35.97],[-5.2,36.05],[-4.8,36.1]],
  '마젤란 해협': [[-68.2,-52.3],[-69.4,-52.5],[-70.2,-52.75],[-70.85,-53.2],[-71.1,-53.75],[-71.35,-53.9],[-72.3,-53.6],[-73.3,-53.05],[-74.3,-52.75],[-75,-52.6]],
  '싱가포르 해협': [[103.4,1.15],[103.8,1.2],[104.2,1.25],[104.5,1.3]],
  '순다 해협': [[105.6,-6.6],[105.75,-6.1],[105.85,-5.85],[106.0,-5.7]],
  '말라카 해협 남단': [[101.0,2.6],[101.8,2.0],[102.6,1.5],[103.4,1.15]],
  '바브엘만데브 해협': [[43.2,12.8],[43.4,12.6],[43.6,12.4]],
  '호르무즈 해협': [[56.2,26.6],[56.5,26.3],[56.8,26.0],[57.2,25.7]],
  '도버 해협': [[1.2,50.9],[1.5,51.05],[1.8,51.2]],
  '산베르나르디노 해협': [[124.9,12.65],[124.4,12.55],[124.0,12.6],[123.6,12.8]],
  '베르데섬 수로': [[122.0,13.2],[121.4,13.6],[120.9,13.7],[120.6,13.85],[120.6,14.3],[120.75,14.5]],
  '보스포루스 대신 · 메시나 해협': [[15.55,38.1],[15.62,38.2],[15.65,38.3]],
  '타구스 하구': [[-9.5,38.65],[-9.25,38.69],[-9.13,38.7]],
  '과달키비르 하구': [[-6.45,36.78],[-6.36,36.8]],
};
const cellOf = (lon, lat) => { const r = Math.floor((lat - LAT0) / RES), c = ((Math.floor((lon - LON0) / RES) % W) + W) % W; return r < 0 || r >= H ? -1 : r * W + c; };
const ll = i => [LON0 + (i % W + 0.5) * RES, LAT0 + (Math.floor(i / W) + 0.5) * RES];
for (const pts of Object.values(CHANNELS)) for (let k = 1; k < pts.length; k++){
  const [a, b] = [pts[k - 1], pts[k]]; const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.02);
  for (let t = 0; t <= n; t++){ const i = cellOf(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n); if (i >= 0) land[i] = 0; }
}
// 북극해를 닫는다: 북서항로(캐나다 북극 제도 · 패리 해협 74°N)와 베링 해협 너머 북극해는 대항해시대에 아무도 지나가지 못했다
// (북서항로 첫 완주는 아문센 1903~06, 북동항로는 노르덴셸드 1878~79). 열어 두면 A* 가 런던→북극→태평양 같은 지름길을 고른다.
// 66°N 위는 모두 닫고, 노르웨이 연안(서경 30°~동경 60°)만 72°N 까지 남긴다.
{
  let closed = 0;
  for (let i = 0; i < W * H; i++){ if (land[i]) continue; const [lon, lat] = ll(i);
    if (lat > 66 && !(lon > -30 && lon < 60 && lat < 72)){ land[i] = 1; closed++; } }
  console.log(`북극해 ${closed}칸 닫음 (북서·북동항로 차단)`);
}
// 대양과 이어진 바다만 남긴다: 카스피해·오대호·흑해 같은 갇힌 물은 '육지'로 보아 배가 들어가지 않게 하고,
// 항구를 붙일 때도 대양 쪽 칸에만 붙게 한다 (도달 불가 길찾기를 원천 차단 → 길찾기 시간이 일정해진다)
{
  const comp = new Int32Array(W * H).fill(-1), sizes = []; let id = 0;
  for (let s = 0; s < W * H; s++){ if (land[s] || comp[s] >= 0) continue; let n = 0; const q = [s]; comp[s] = id;
    while (q.length){ const u = q.pop(); n++; const r = Math.floor(u / W), c = u % W;
      for (const [dr, dc] of [[1,0],[-1,0],[0,1],[0,-1]]){ const rr = r + dr; if (rr < 0 || rr >= H) continue; const v = rr * W + (c + dc + W) % W; if (!land[v] && comp[v] < 0){ comp[v] = id; q.push(v); } } }
    sizes.push(n); id++; }
  const main = sizes.indexOf(Math.max(...sizes)); let shut = 0;
  for (let i = 0; i < W * H; i++) if (!land[i] && comp[i] !== main){ land[i] = 1; shut++; }
  console.log(`바다 덩어리 ${sizes.length}개 중 대양 하나만 남김 (갇힌 물 ${shut}칸 닫음)`);
}
const sea = i => i >= 0 && !land[i];
const nearLand = new Uint8Array(W * H);
for (let i = 0; i < W * H; i++){ if (land[i]) continue; const r = Math.floor(i / W), c = i % W;
  outer: for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++){ const rr = r + dr; if (rr < 0 || rr >= H) continue; if (land[rr * W + (c + dc + W) % W]){ nearLand[i] = 1; break outer; } } }

/* ── 4. 길찾기 (A*, 8방향, 좌우는 이어짐) ── */
const R2D = Math.PI / 180;
const km = (a, b) => { const dl = ((b[0] - a[0] + 540) % 360 - 180) * R2D, p1 = a[1] * R2D, p2 = b[1] * R2D; return 6371 * 2 * Math.asin(Math.min(1, Math.sqrt(Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2))); };
function nearestSea(lon, lat, maxR = 16){
  const c0 = cellOf(lon, lat); if (sea(c0)) return c0;
  const r0 = Math.floor(c0 / W), cc = c0 % W; let best = -1, bd = 1e9;
  for (let rad = 1; rad <= maxR && best < 0; rad++) for (let dr = -rad; dr <= rad; dr++) for (let dc = -rad; dc <= rad; dc++){
    if (Math.max(Math.abs(dr), Math.abs(dc)) !== rad) continue; const rr = r0 + dr; if (rr < 0 || rr >= H) continue; const i = rr * W + (cc + dc + W) % W;
    if (sea(i)){ const d = km([lon, lat], ll(i)); if (d < bd){ bd = d; best = i; } } }
  return best;
}
const gS = new Float64Array(W * H), came = new Int32Array(W * H), seen = new Uint32Array(W * H); let gen = 0;
function astar(s, g){
  gen++; const heap = []; const push = (f, i) => { heap.push([f, i]); let k = heap.length - 1; while (k){ const p = (k - 1) >> 1; if (heap[p][0] <= heap[k][0]) break; [heap[p], heap[k]] = [heap[k], heap[p]]; k = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length){ heap[0] = last; let k = 0; for (;;){ const l = 2 * k + 1, r = l + 1; let m = k; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === k) break; [heap[m], heap[k]] = [heap[k], heap[m]]; k = m; } } return top; };
  const G = ll(g); seen[s] = gen; gS[s] = 0; came[s] = -1; push(km(ll(s), G), s);
  while (heap.length){
    const [, u] = pop(); if (u === g) break;
    const r = Math.floor(u / W), c = u % W, lat = LAT0 + (r + 0.5) * RES, cx = Math.cos(lat * R2D) * RES * 111.2, cy = RES * 111.2;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++){
      if (!dr && !dc) continue; const rr = r + dr; if (rr < 0 || rr >= H) continue; const v = rr * W + (c + dc + W) % W; if (!sea(v)) continue;
      if (dr && dc && (!sea(r * W + (c + dc + W) % W) || !sea(rr * W + c))) continue; // 대각선으로 육지 모서리를 스치지 않게
      const w = Math.hypot(dc * cx, dr * cy) * (nearLand[v] ? 1.35 : 1); const nd = gS[u] + w;
      if (seen[v] !== gen || nd < gS[v]){ seen[v] = gen; gS[v] = nd; came[v] = u; push(nd + km(ll(v), G), v); }
    }
  }
  if (seen[g] !== gen) return null;
  const out = []; for (let k = g; k !== -1; k = came[k]) out.push(k); return out.reverse();
}
function los(a, b){ // 선분이 지나는 칸을 빠짐없이 따라가며(격자 DDA, 모서리를 지나면 양옆 칸도) 모두 바다인지 본다
  let lo2 = b[0]; if (Math.abs(lo2 - a[0]) > 180) lo2 += lo2 < a[0] ? 360 : -360;
  const x0 = (a[0] - LON0) / RES, y0 = (a[1] - LAT0) / RES, x1 = (lo2 - LON0) / RES, y1 = (b[1] - LAT0) / RES;
  let cx = Math.floor(x0), cy = Math.floor(y0); const ex = Math.floor(x1), ey = Math.floor(y1), dx = x1 - x0, dy = y1 - y0, sx = Math.sign(dx), sy = Math.sign(dy);
  let tx = dx ? ((sx > 0 ? cx + 1 : cx) - x0) / dx : Infinity, ty = dy ? ((sy > 0 ? cy + 1 : cy) - y0) / dy : Infinity; const ddx = dx ? Math.abs(1 / dx) : Infinity, ddy = dy ? Math.abs(1 / dy) : Infinity;
  const ok = (x, y) => y >= 0 && y < H && sea(y * W + ((x % W) + W) % W);
  if (!ok(cx, cy)) return false;
  for (let guard = 0; (cx !== ex || cy !== ey) && guard < 200000; guard++){
    if (Math.abs(tx - ty) < 1e-9){ if (!ok(cx + sx, cy) || !ok(cx, cy + sy)) return false; cx += sx; cy += sy; tx += ddx; ty += ddy; }
    else if (tx < ty){ cx += sx; tx += ddx; } else { cy += sy; ty += ddy; }
    if (!ok(cx, cy)) return false;
  }
  return true;
}
function smooth(pts){ // 줄 당기기: 바다로만 이어지는 한 가장 멀리 건너뛴다
  const out = [pts[0]]; let i = 0;
  while (i < pts.length - 1){ let j = Math.min(pts.length - 1, i + 400); while (j > i + 1 && !los(pts[i], pts[j])) j--; out.push(pts[j]); i = j; }
  return out;
}

/* ── 5. 역사 항로 제어점 (사료의 기항지·경유지) ── */
const ROUTES = [
  {name:'인도 항로 · 가는 길 (볼타 두 마르 → 모잠비크 해협)', color:'#e8b04f', dir:true,
   note:'리스본 → 카나리아 앞바다(기항 안 함) → 카보베르데(첫 기항지) → 무풍대에서 남서로 흘러 페르남부쿠·산투아고스티뉴곶 앞 → 브라질 해류를 타고 남회귀선 부근까지 → 편서풍으로 동진, 남아프리카 해안에서 멀찍이 → 모잠비크 해협 한가운데(바사스다인디아·유로파 사이) → 모잠비크섬 → 적도 남쪽에서 남서 계절풍 타고 동진 → 안제디바 상륙 → 고아 → 코친',
   pts:[[-9.4,38.6],[-16.5,29.5],[-23.6,14.8],[-24,5],[-30,-3],[-34.3,-8.4],[-36.5,-16],[-34,-22.5],[-15,-31],[5,-36],[20,-37.5],[32,-32],[36.6,-25.2],[40,-22],[40.9,-15.2],[42.6,-8],[44.5,-3.5],[55,3],[65,10],[73.8,14.75],[73.6,15.45],[74.4,12.8],[76.1,9.95]]},
  {name:'인도 항로 · 돌아오는 길', color:'#c99a45', dir:true,
   note:'코친·고아 → 모잠비크 해협(해안 가까이) → 아굴라스 해류로 희망봉 → 남동 무역풍, 세인트헬레나·어센션 서쪽 → 무풍대 → 북상해 아소르스 → 편서풍으로 리스본. (1525–1579년에는 마다가스카르 동쪽 바깥 항로가 의무)',
   pts:[[76.1,9.95],[72,6],[60,-6],[44,-12],[40.9,-15.6],[37,-22],[33,-29],[27,-34.5],[19,-35.6],[12,-30],[-8,-16],[-16,-8],[-25,3],[-35,20],[-38,32],[-27.5,38.6],[-15,38.8],[-9.4,38.6]]},
  {name:'서인도 항로 · 갈 때 (누에바에스파냐 선단)', color:'#e07a5a', dir:true,
   note:'산루카르(과달키비르 하구) → 아프리카 해안을 따라 카나리아 기항 → 무역풍으로 대서양 횡단 → 소앤틸리스(도미니카 부근) 진입 → 푸에르토리코·히스파니올라 남쪽 → 유카탄 해협 → 베라크루스. 소앤틸리스 진입 지점은 사료로 확인하지 못한 근사',
   pts:[[-6.4,36.78],[-9,35],[-15.8,28.3],[-35,20],[-55,16],[-61.25,15.35],[-66,17.5],[-71,17.1],[-77,18.2],[-82,19.8],[-85.4,21.7],[-90,22.3],[-96,19.3]]},
  {name:'서인도 항로 · 갈 때 (티에라피르메 선단)', color:'#d0694c', dir:true,
   note:'소앤틸리스에서 갈라져 카르타헤나 → 포르토벨로(페루 은이 지협을 건너 오는 곳)',
   pts:[[-61.25,15.35],[-66,13.5],[-72,12.5],[-75.6,10.5],[-77.5,10],[-79.65,9.6]]},
  {name:'선단 집결 · 아바나로', color:'#e89a7a', dir:true,
   note:'포르토벨로 → 카르타헤나 → 유카탄 해협 → 아바나 / 베라크루스 → 아바나. 두 선단이 아바나에서 모여 귀환 준비',
   pts:[[-79.65,9.6],[-75.6,10.5],[-79,14],[-84.5,20.5],[-84.9,22],[-82.4,23.2]]},
  {name:'보물선단 귀환 (플로리다 해협 → 걸프 해류)', color:'#f0a080', dir:true,
   note:'아바나 → 플로리다 해협 → 걸프 해류를 타고 플로리다 해안을 따라 북상 → 에스파냐 위도(37~38°N)에서 동쪽으로 → 아소르스 → 산비센테곶 → 산루카르',
   pts:[[-82.4,23.2],[-80.4,24.35],[-79.8,26.5],[-79.7,30],[-75.3,34.6],[-60,37.4],[-40,38],[-27.2,38.7],[-15,37.3],[-9.4,36.95],[-6.4,36.78]]},
  {name:'마닐라 갈레온 · 서쪽으로', color:'#7fc4dc', dir:true,
   note:'아카풀코 → 북위 13° 안팎을 따라 서진(하와이 남쪽) → 괌 기항(1668년부터 의무) → 사마르의 산베르나르디노 해협 → 마닐라만 카비테',
   pts:[[-99.9,16.75],[-105,14],[-130,13],[-157,13.3],[180,13.4],[160,13.4],[144.6,13.4],[130,12.8],[124.9,12.6],[121.4,13.6],[120.75,14.5]]},
  {name:'마닐라 갈레온 · 동쪽으로 (구로시오)', color:'#5aa9c8', dir:true,
   note:'카비테 → 산베르나르디노 해협 → 일본 동쪽 북위 38° 부근까지 북상 → 편서풍으로 북태평양 횡단 → 콘셉시온곶~산루카스곶 사이 해안에 상륙 → 해안을 따라 남하해 아카풀코',
   pts:[[120.75,14.5],[121.4,13.6],[124.9,12.6],[128,15],[133,22],[141,33],[146,37.5],[160,38.5],[180,38.5],[-160,37.5],[-140,36],[-121,34.2],[-117.5,31.5],[-112.5,25.5],[-109.6,22.5],[-105.5,18.8],[-99.9,16.75]]},
  {name:'마젤란 원정 (1519–21)', color:'#b9a0e6', dir:true,
   note:'산루카르(1519.9.20) → 카나리아 → 아프리카 해안을 따라 시에라리온 부근 → 대서양 횡단, 산투아고스티뉴곶(11.29) → 리우데자네이루(12.13) → 라플라타(1.12) → 산훌리안 월동(3.31) → 산타크루스 → 비르헤네스곶에서 해협 진입(10.21) → 태평양(11.28) → 칠레 연안 북상 후 북서진 → 푸카푸카(1.24) → 괌(3.6) → 호몬혼(3.16) → 리마사와 → 세부(4.7) → 막탄(4.27). 태평양 횡단 중간점은 근사',
   pts:[[-6.4,36.78],[-15.8,28.3],[-17.5,20],[-14,8.5],[-25,1],[-34.6,-8.4],[-38.5,-15],[-43.15,-23.05],[-48,-28],[-54,-34.8],[-57,-35.3],[-56.5,-37.5],[-62,-40.5],[-64.5,-42],[-65.5,-45],[-67.5,-49.3],[-68.2,-50.15],[-68.2,-52.3],[-70.85,-53.2],[-71.35,-53.9],[-73.3,-53.05],[-75,-52.6],[-77,-47],[-78,-33],[-110,-22],[-138.8,-15],[-151.8,-11.4],[-162,0],[-175,10],[160,13],[144.6,13.4],[125.8,10.8],[125.1,9.9],[124.4,10.35],[124.05,10.3]]},
  {name:'엘카노 귀환 (빅토리아호, 1522)', color:'#9d8ad0', dir:true,
   note:'티도레 → 반다해 → 티모르(1.25) → 포르투갈 항로를 피해 인도양 남쪽 → 희망봉 통과(5.22) → 카보베르데 산티아구(7.9) → 산루카르(9.6)',
   pts:[[127.4,0.6],[126,-4],[124.5,-8.3],[124,-10.6],[110,-20],[80,-35],[40,-38],[19,-35.8],[8,-24],[-10,-8],[-20,8],[-23.5,14.7],[-24,25],[-15,35],[-9.5,36.9],[-6.4,36.78]]},
  {name:'브라우어 항로 (네덜란드, 1611~)', color:'#86c788', dir:true,
   note:'희망봉 → 남쪽 포효하는 40도대로 내려가 편서풍으로 동진(남위 36~40°) → 암스테르담섬·생폴섬을 보고 북쪽으로 → 순다 해협 → 바타비아. 너무 동쪽까지 가면 오스트레일리아 서해안에 좌초',
   pts:[[18.4,-34.6],[20,-38],[45,-38.5],[77.5,-38.4],[92,-28],[102,-14],[105.5,-7.2],[105.75,-6.1],[106.8,-5.95]]},
  {name:'몬순 무역로 (믈라카 – 캘리컷 – 호르무즈)', color:'#d8cfa2', dir:false,
   note:'아시아 상인들의 일반 무역로를 단순화한 것. 특정 항해 기록이 아니라 계절풍 무역의 대표 경로',
   pts:[[102.2,2.1],[100,4],[97,6],[86,6],[80.6,5.6],[77,7.5],[75.6,11.3],[66,17],[60,22.6],[57,25.4],[56.45,26.9]]},
  {name:'남해 은 항로 (카야오 – 파나마)', color:'#cfd6dc', dir:false,
   note:'페루 부왕령의 은을 카야오(리마)에서 파나마로 실어 나른 남해 함대의 연안 항로',
   pts:[[-77.2,-12.05],[-79,-9],[-81.3,-6],[-81.3,-2.5],[-80.5,0],[-80,3],[-79.5,7],[-79.5,8.9]]},
];
const routesOut = [];
const corridor = new Map(); // 칸 → [항로 번호, dx, dy]
ROUTES.forEach((rt, ri) => {
  // 제어점 사이를 따로 잇고 따로 펴서, 사료의 경유지를 반드시 지나게 한다
  let raw = [];
  for (let k = 1; k < rt.pts.length; k++){
    const s = nearestSea(...rt.pts[k - 1]), g = nearestSea(...rt.pts[k]);
    if (s < 0 || g < 0){ console.error(`  ! ${rt.name} 제어점 ${k} 근처에 바다 없음`); continue; }
    const p = astar(s, g); if (!p){ console.error(`  ! ${rt.name} 구간 ${k} 길 없음`); continue; }
    const seg = smooth(p.map(ll)); raw = raw.concat(raw.length ? seg.slice(1) : seg);
  }
  const pts = raw.map(([a, b]) => [+a.toFixed(3), +b.toFixed(3)]);
  routesOut.push({name:rt.name, color:rt.color, dir:rt.dir, note:rt.note, ctrl:rt.pts, pts});
  for (let k = 1; k < pts.length; k++){ // 항로 둘레 칸에 방향을 적어 둔다 (배가 그 방향으로 갈 때 싸게)
    let [lo1, la1] = pts[k - 1], [lo2, la2] = pts[k]; if (Math.abs(lo2 - lo1) > 180) lo2 += lo2 < lo1 ? 360 : -360;
    const dx = (lo2 - lo1) * Math.cos((la1 + la2) / 2 * R2D), dy = la2 - la1, L = Math.hypot(dx, dy) || 1, n = Math.ceil(Math.hypot(lo2 - lo1, la2 - la1) / 0.1);
    for (let t = 0; t <= n; t++){ let lo = lo1 + (lo2 - lo1) * t / n; if (lo > 180) lo -= 360; const la = la1 + (la2 - la1) * t / n; const c0 = cellOf(lo, la); if (c0 < 0) continue;
      const r = Math.floor(c0 / W), c = c0 % W;
      for (let dr = -2; dr <= 2; dr++) for (let dc = -2; dc <= 2; dc++){ const rr = r + dr; if (rr < 0 || rr >= H) continue; const i = rr * W + (c + dc + W) % W; if (sea(i) && !corridor.has(i)) corridor.set(i, [ri, Math.round(dx / L * 100), Math.round(dy / L * 100)]); } }
  }
});

/* ── 6. 저장 + 자체 점검(항로 선분이 바다로만 가는지) ── */
const bits = new Uint8Array(Math.ceil(W * H / 8)); for (let i = 0; i < W * H; i++) if (!land[i]) bits[i >> 3] |= 1 << (i & 7);
let bad = 0; for (const r of routesOut) for (let k = 1; k < r.pts.length; k++) if (!los(r.pts[k - 1], r.pts[k])){ bad++; console.error(`  ! ${r.name} 선분 ${k} 육지 통과`); }
const out = {grid:{res:RES, lon0:LON0, lat0:LAT0, w:W, h:H}, mask:Buffer.from(bits).toString('base64'), routes:routesOut,
  corridor:[...corridor].flat(), channels:Object.keys(CHANNELS)};
fs.writeFileSync(path.join(ROOT, 'data/seagrid.json'), JSON.stringify(out));
const seaCells = W * H - land.reduce((s, v) => s + v, 0);
console.log(`격자 ${W}×${H} (${RES}°) · 바다 ${(seaCells / (W * H) * 100).toFixed(1)}% · 해협 ${Object.keys(CHANNELS).length}곳 개방 · 역사 항로 ${routesOut.length}개 (제어점 ${ROUTES.reduce((s, r) => s + r.pts.length, 0)}개 → 선분 ${routesOut.reduce((s, r) => s + r.pts.length - 1, 0)}개) · 육지 통과 선분 ${bad}개 · 항로 띠 ${corridor.size}칸`);
