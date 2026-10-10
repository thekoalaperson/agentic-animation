// Pure controller tests: no browser, network, media decode or file URL navigation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, process.argv[2] || 'sound_revision_review_template.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
new Function(script);
const controls = script.slice(script.indexOf('async function play()'), script.indexOf('\nfunction updateTime()'));
const keyboard = script.slice(script.indexOf("document.addEventListener('keydown',e=>{"), script.indexOf("\n$('export').onclick"));
function harness() {
  const ui = {status: {}, play: {}};
  const player = {currentTime: 37, paused: false, duration: 70, readyState: 1,
    pause() { this.paused = true; },
    play() { this.paused = false; return Promise.resolve(); },
    load() { this.currentTime = 0; this.paused = true; this.readyState = 0; }};
  const sandbox = {player, $: id => ui[id], current: () => ({variants: [
    {label: 'Old', url: 'old.mp4'}, {label: 'New A', url: 'a.mp4'}, {label: 'New B', url: 'b.mp4'}]}),
    document: {querySelectorAll: () => [], addEventListener: (_, fn) => {sandbox.keydown = fn;}},
    updateTime() {}, renderScene() {sandbox.changedScene = true;}, sceneIndex: 0};
  vm.createContext(sandbox);
  vm.runInContext('let variantIndex=1, loadToken=0, playbackTarget=null;\n' + controls + keyboard, sandbox);
  return {sandbox, player, ui, run: code => vm.runInContext(code, sandbox)};
}
function event(target, code, key = code, extra = {}) {
  return {target, code, key, ctrlKey: false, altKey: false, metaKey: false,
    preventDefault() {this.prevented = true;}, stopPropagation() {this.stopped = true;}, ...extra};
}
{
  const h = harness();
  h.run('chooseVariant(2)');
  const stale = h.player.onloadedmetadata;
  h.run('chooseVariant(0)');
  assert.equal(h.run('playbackTarget.time'), 37);
  assert.equal(h.run('playbackTarget.playing'), true);
  h.player.readyState = 1;
  stale();
  assert.equal(h.player.currentTime, 0, 'stale metadata must not seek');
  h.player.onloadedmetadata();
  assert.equal(h.player.currentTime, 37);
  assert.equal(h.player.paused, false);
}
{
  const h = harness();
  h.run('chooseVariant(2,true,48)');
  const oldSceneLoad = h.player.onloadedmetadata;
  h.run('chooseVariant(1,false)');
  h.player.readyState = 1;
  oldSceneLoad();
  h.player.onloadedmetadata();
  assert.equal(h.player.currentTime, 0, 'old wave seek must not leak into new scene');
  assert.equal(h.player.paused, true);
}
{
  const h = harness();
  h.run('chooseVariant(2)');
  h.run('toggle()');
  assert.equal(h.run('playbackTarget.playing'), false);
  h.player.readyState = 1;
  h.player.onloadedmetadata();
  assert.equal(h.player.paused, true, 'pause during loading must persist');
}
{
  const h = harness();
  const raw = {tagName: 'AUDIO', paused: true, play() {this.paused = false;return Promise.resolve();}, pause() {this.paused = true;}};
  h.sandbox.keydown(event(raw, 'Space', ' '));
  assert.equal(raw.paused, false, 'Space should play the focused raw sound');
  assert.equal(h.player.currentTime, 37);
  const arrow = event(raw, 'ArrowLeft');
  h.sandbox.keydown(arrow);
  assert.equal(arrow.prevented, undefined, 'raw audio arrows keep native behavior');
  const notes = {tagName: 'TEXTAREA', matches: () => true, isContentEditable: false, blur() {this.blurred = true;}};
  const space = event(notes, 'Space', ' ');
  h.sandbox.keydown(space);
  assert.equal(space.prevented, undefined, 'notes keep ordinary spaces');
  const left = event(notes, 'ArrowLeft');
  h.sandbox.keydown(left);
  assert.equal(left.prevented, undefined, 'notes keep ordinary arrows');
  h.sandbox.keydown(event(notes, 'Space', ' ', {ctrlKey: true}));
  assert.equal(h.player.paused, true, 'Ctrl+Space controls film from notes');
  h.sandbox.keydown(event(notes, 'Escape'));
  assert.equal(notes.blurred, true);
}
{
  const bindingLine = script.split('\n').find(l => l.startsWith('const reviewBinding='));
  const data = {planSha256: 'plan', musicManifestSha256: 'manifest', scenes: [{variants: [{sha256:'take1'}]}], sounds: [{sha256:'fx1'}]};
  const binding = d => vm.runInNewContext(bindingLine+'\nreviewBinding', {DATA:d});
  const first = binding(data);
  data.scenes[0].variants[0].sha256 = 'regenerated';
  assert.notEqual(binding(data), first, 'a regenerated take cannot inherit old approvals');
  const second = binding(data);data.sounds[0].sha256='newfx';
  assert.notEqual(binding(data), second, 'changed effects create a new review binding');
}
console.log('PASS: rapid switching, stale seeks, pending pause, raw audio Space, native note typing, Ctrl+Space, Escape and hash-bound feedback.');
