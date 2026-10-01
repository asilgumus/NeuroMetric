// Execute the actual inline dashboard script with a minimal DOM, for every case.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'report.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element {
  constructor() { this.children = []; this.attributes = {}; this.listeners = {}; this.style = {setProperty(){}}; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  set textContent(value) { this.text = value; this.children = []; }
  get textContent() { return this.text || ''; }
  appendChild(element) { this.children.push(element); }
  querySelector() { return new Element(); }
  setAttribute(key, value) { this.attributes[key] = value; }
}
const exactSandbox = {window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root, 'assets/exact-hit-cases.js'), 'utf8'), exactSandbox);
const exactCases = exactSandbox.window.NEUROMETRIC_EXACT_CASES;
vm.runInNewContext(fs.readFileSync(path.join(root, 'assets/oasis7-case.js'), 'utf8'), exactSandbox);
exactCases.oasis7 = exactSandbox.window.NEUROMETRIC_OASIS7_CASE;
for (const key of [...Object.keys(exactCases), 'a', 'b', 'c', 'd', 'invalid', 'oasis30', 'oasis2']) {
  const effective = Object.hasOwn(exactCases, key) ? key : 'ixi361';
  const c = exactCases[effective];
  const elements = new Map();
  const document = {
    getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
    createElement() { return new Element(); },
    querySelectorAll() { return []; }
  };
  const sandbox = {document, window:{location:{search:'?case='+key}}, URLSearchParams};
  for (const asset of ['oasis30-longitudinal.js','oasis30-regional.js','oasis30-regional-candidate.js','oasis2-longitudinal-reference.js','exact-hit-cases.js','selected-regional-cases.js','selected-regional-review.js','oasis7-case.js','oasis7-longitudinal.js']) {
    vm.runInNewContext(fs.readFileSync(path.join(root,'assets',asset),'utf8'),sandbox);
  }
  vm.runInNewContext(script,sandbox);
  assert.equal(elements.get('dPred').innerHTML,c.pred+'<span class="pad-unit">yr</span>');
  assert.ok(elements.get('dPad').innerHTML.startsWith(c.gap));
  assert.equal(elements.get('realHeat').hidden,false);
  assert.equal(elements.get('realHeat').style.display,'block');
  assert.equal(elements.get('realHeat').src,'assets/'+c.heatPrefix+'-axial.png');
  assert.ok(elements.get('realHeat').alt.startsWith(c.patientId));
  assert.equal(elements.get('attributionLegend').hidden,false);
  assert.equal(elements.get('realFollowUp').hidden,true);
  const regional=sandbox.window.NEUROMETRIC_SELECTED_REGIONS[c.patientId];
  const measured=regional && ['visual_review_pending','technically_reviewed'].includes(regional.state);
  assert.equal(elements.get('dBars').children.length,measured ? 5 : 1);
  if(measured) {
    assert.equal(elements.get('regionalVisual').hidden,false);
    assert.equal(elements.get('dBars').children[0].children[1].textContent,Number(regional.regions[0].volume_ml).toFixed(2)+' mL');
  } else assert.ok(elements.get('dBars').children[0].textContent.includes(c.patientId));
  const paired=effective==='oasis7' && sandbox.window.NEUROMETRIC_OASIS7_LONGITUDINAL;
  assert.equal(elements.get('dTrend').children.length,paired ? 2 : 1);
  assert.equal(elements.get('trajectoryChart').innerHTML.includes('polyline'),Boolean(paired));
  elements.get('viewNext').listeners.click();
  assert.equal(elements.get('realHeat').src,'assets/'+c.heatPrefix+'-coronal.png');
  elements.get('viewPrev').listeners.click();
  elements.get('mriViewer').listeners.keydown({key:'ArrowLeft',preventDefault(){}});
  assert.equal(elements.get('realHeat').src,'assets/'+c.heatPrefix+'-sagittal.png');
}
const landing = fs.readFileSync(path.join(root,'index.html'),'utf8');
const select = landing.match(/<select id="caseSel"[\s\S]*?<\/select>/)[0];
assert.equal((select.match(/<option/g)||[]).length,5);
assert.ok(!/oasis2"|oasis30|NC-/.test(select));
assert.ok(select.includes('oasis7'));
console.log('Four real exact-hit cases, own-patient heatmaps and removed-case fallbacks passed.');
