// Offline behavior tests; these do not claim a live browser playback test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
let page,html;
if(process.argv[2]){page=path.resolve(process.argv[2]);html=fs.readFileSync(page,'utf8');}
else{
 const os=require('node:os');
 const fixture=fs.mkdtempSync(path.join(os.tmpdir(),'sfx-controller-'));
 process.on('exit',()=>fs.rmSync(fixture,{recursive:true,force:true}));
 page=path.join(fixture,'review.html');
 const families=[0,1].map(f=>({id:'family-'+f,title:'Family '+f,category:'authored',intent:'Synthetic controller fixture',shots:[f+1],placements:[],candidates:[0,1].map(t=>{
  const id='sound-'+f+'-t'+t,url=id+'.wav';fs.writeFileSync(path.join(fixture,url),'synthetic-link-fixture');
  return {key:id,id,hash:String(f*2+t+1).repeat(64),family:'family-'+f,path:url,url,model:'synthetic',batches:['fixture'],prompt:'Synthetic test sound',duration:2,peaks:[0,.2,.1],used:false,newSource:true,label:'Unreviewed'};
 })}));
 const data={episode:'DEMO',episodeCode:'DEMO',standalone:true,newCandidates:4,totalCandidates:4,families,currentVerdict:'unreviewed',designStatus:'Controller fixture only'};
 html=fs.readFileSync(path.join(__dirname,'sfx_debug_template.html'),'utf8').replace('/*__DATA__*/null',JSON.stringify(data));
}
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const listeners={},elements=new Map();
class El{
 constructor(tag='DIV'){this.tagName=tag;this.value='';this.children=[];this.dataset={};this.attrs={};this.paused=true;this.currentTime=0;this.duration=10;}
 append(...c){this.children.push(...c);}replaceChildren(){this.children=[];}setAttribute(k,v){this.attrs[k]=v;}removeAttribute(k){delete this[k];}
 getBoundingClientRect(){return{width:700,height:128,left:0};}getContext(){return{scale(){},beginPath(){},moveTo(){},lineTo(){},stroke(){}};}
 pause(){this.paused=true;this.onpause?.();}async play(){this.paused=false;this.onplay?.();}load(){this.currentTime=0;}focus(){}blur(){}click(){}
}
for(const match of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"/g))elements.set(match[2],new El(match[1].toUpperCase()));
elements.get('filter').value='used';
const verdicts=['reject','maybe','keep','unreviewed'].map(v=>{let e=new El('BUTTON');e.dataset.v=v;return e;});
let saved='',exported='';const context=vm.createContext({document:{getElementById:id=>elements.get(id),createElement:t=>new El(t.toUpperCase()),createTextNode:s=>s,querySelectorAll:q=>q==='[data-v]'?verdicts:[],addEventListener:(k,f)=>listeners[k]=f},window:{devicePixelRatio:1,addEventListener(){}},localStorage:{getItem(){return saved;},setItem(k,v){saved=v;}},requestAnimationFrame:f=>f(),setTimeout:()=>0,URL:{createObjectURL:()=>'',revokeObjectURL(){}},Blob:class{constructor(parts){exported=parts.join('');}},console});
vm.runInContext(script,context);
const run=s=>vm.runInContext(s,context);
const hasNew=run('hasNew');
const focus=run('Boolean(DATA.focusBatch)');
const standalone=run('Boolean(DATA.standalone)');
const diverse=run('DATA.focusBatch === "sfx-diversity-v4"');
const focusCount=diverse?12:16;
assert.equal(run('visible.length'),standalone?run('DATA.families.length'):focus?focusCount:hasNew?41:34);assert.equal(run('DATA.totalCandidates'),standalone?run('DATA.newCandidates'):diverse?251:focus?227:hasNew?195:111);
if(hasNew){if(!standalone){assert.equal(run('DATA.newCandidates'),diverse?24:focus?32:84);assert.equal(run('candidates().length'),2);}assert.equal(run('candidate.newSource'),true);}
if(run('DATA.plannedMythicCandidates')){
 assert.equal(run('DATA.plannedMythicCandidates'),32);
 assert.equal(run('DATA.families.filter(f=>f.nextDesign).length'),16);
 run('selectFamily(DATA.families.find(f=>f.nextDesign))');
 assert.equal(elements.get('design').hidden,false);
 assert.match(elements.get('design-data').textContent,/not generated/);
 assert.match(elements.get('design-data').textContent,/scoreMeasurement/);
 run('selectFamily(visible[0])');
}
const all=run('JSON.stringify(DATA.families.flatMap(f=>f.candidates))');
for(const c of JSON.parse(all)){for(const a of [c,c.contextAudio,c.baselineAudio].filter(Boolean)){assert(fs.existsSync(path.resolve(path.dirname(page),decodeURIComponent(a.url))));assert(a.peaks.length>0);assert(a.duration>0);}}
if(focus&&!standalone){
 assert.equal(run('DATA.families.flatMap(f=>f.candidates).filter(c=>c.contextAudio).length'),diverse?56:32);
 const raw=elements.get('audio').src;
 elements.get('audition').value='context';elements.get('audition').onchange();assert.notEqual(elements.get('audio').src,raw);
 assert.equal(run('auditionAsset().duration'),14);
 const mixed=elements.get('audio').src;
 elements.get('audition').value='baseline';elements.get('audition').onchange();assert.notEqual(elements.get('audio').src,mixed);
 elements.get('audition').value='raw';elements.get('audition').onchange();assert.equal(elements.get('audio').src,raw);
 assert.match(elements.get('design-data').textContent,/generated/);
}
assert.equal(run('candidate.used'),!hasNew);
const initial=run('candidate.key');
function key(k,tag='BUTTON',mods={}){let prevented=false;listeners.keydown({key:k,code:k===' '?'Space':k,target:new El(tag),preventDefault(){prevented=true;},...mods});return prevented;}
assert(key('ArrowRight'));if(run('candidates().length')>1)assert.notEqual(run('candidate.key'),initial);
assert(key(' '));assert.equal(elements.get('audio').paused,false);
assert(key('ArrowDown'));assert.equal(elements.get('audio').paused,true);
const before=run('candidate.key');assert.equal(key('ArrowRight','TEXTAREA'),false);assert.equal(run('candidate.key'),before);
assert.equal(key(' ','TEXTAREA'),false);
assert(key('ArrowRight','TEXTAREA',{altKey:true}));if(run('candidates().length')>1)assert.notEqual(run('candidate.key'),before);
elements.get('notes').oninput({target:{value:'Too much hiss, no convincing cloth.'}});
verdicts[0].onclick();assert.match(saved,/Too much hiss/);assert.match(saved,/reject/);
const reviewed=run('candidate.key');run('move("take",1);move("take",-1)');assert.equal(run('candidate.key'),reviewed);assert.equal(elements.get('notes').value,'Too much hiss, no convincing cloth.');
if(hasNew&&!standalone){
 elements.get('legacy').checked=true;elements.get('legacy').onchange();
 assert(run('candidates().some(c=>!c.newSource)'));
 run('selectTake(candidates().find(c=>!c.newSource))');
 assert.equal(run('!!candidate.newSource'),false);
 elements.get('notes').oninput({target:{value:'Old source remains rejected.'}});verdicts[0].onclick();
 elements.get('legacy').checked=false;elements.get('legacy').onchange();
 assert.equal(run('candidate.newSource'),true);
 assert.equal(run('candidates().some(c=>!c.newSource)'),false);
 assert(saved.includes('Old source remains rejected.'));
}
elements.get('export').onclick();const out=JSON.parse(exported);
assert.equal(out.schemaVersion,2);assert.equal(out.currentMixVerdict,standalone?'unreviewed':'rejected');
assert(out.reviews.every(r=>r.sha256&&r.model&&r.path));
assert(out.reviews.some(r=>r.notes==='Too much hiss, no convincing cloth.'));
if(hasNew&&!standalone)assert(out.reviews.some(r=>r.notes==='Old source remains rejected.'));
if(focus&&!standalone)assert(out.reviews.filter(r=>r.newSource).every(r=>r.contextPreviewSha256&&r.baselineSha256));
elements.get('filter').value='all';elements.get('filter').onchange();assert.equal(run('visible.length'),standalone?run('DATA.families.length'):focus?focusCount:41);
elements.get('search').value='nonexistent-search-xyz';elements.get('search').oninput();assert.equal(run('visible.length'),0);assert.equal(elements.get('detail').hidden,true);
elements.get('search').value='';elements.get('search').oninput();assert.equal(run('visible.length'),standalone?run('DATA.families.length'):focus?focusCount:41);
console.log(`PASS: syntax, ${run('DATA.totalCandidates')} media links, waveforms, new/old selection, keyboard, typing mode, pause on switch, per-file notes/verdict persistence, JSON export, filters/empty state. Live playback not tested.`);
