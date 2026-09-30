
(() => {
  const $ = id => document.getElementById(id);
  const MEDIAPIPE_VERSION = '0.10.14';
  const fileInput=$('fileInput'), pickerCard=$('pickerCard'), status=$('status'), panel=$('panel'), canvasWrap=$('canvasWrap'), canvas=$('canvas'), ctx=canvas.getContext('2d');
  const segDot=$('segDot'), segStatus=$('segStatus');
  const depthDot=$('depthDot'), depthStatus=$('depthStatus'), depthPreviewCanvas=$('depthPreviewCanvas'), depthPreviewWrap=$('depthPreviewWrap');
  const viewerTools=$('viewerTools'), correctionBanner=$('correctionBanner'), correctionModeLabel=$('correctionModeLabel');
  const processingOverlay=$('processingOverlay'), overlayText=$('overlayText'), overlaySub=$('overlaySub');
  const stickySavebar=$('stickySavebar'), btnSaveSticky=$('btnSaveSticky'), btnShareSticky=$('btnShareSticky');
  const toastEl=$('toast'), toastText=$('toastText');
  const sliderIds=['apertureF','blurStrength','focusDepth','depthOfField','edgeSoftness','bokehSize','bokehIntensity','bokehDensity','bokehFringe','brushSize','filterStrength','bright','sat','contrast','warmth','glow','subjectPop','copyrightOffsetX','copyrightOffsetY','copyrightScale','copyrightOpacity'];
  sliderIds.forEach(id=>{const el=$(id), out=$(id+'Out'); out.textContent=el.value; el.addEventListener('input',()=>out.textContent=el.value);});

  let fullImg=null, fullW=0, fullH=0, workW=0, workH=0, srcCanvas=null;
  let showAfter=true, currentPreset='glow';
  let imageSegmenterInstance=null, autoBackgroundMask=null, segLoading=false, segError=null;
  let depthEstimatorInstance=null, depthEstimatorPromise=null, depthComputePromise=null, depthLoading=false, depthError=null;
  let aiDepthCanvas=null, aiDepthRevision=0, depthAutoFocused=false, depthManualInvert=false, depthOrientationChecked=false;
  const DEPTH_MODEL_ID='onnx-community/depth-anything-v2-small';
  const TRANSFORMERS_JS_URL='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

  // 補正ブラシ用。v4ではこの宣言が抜けていたため、
  // 画像読み込み後の描画がReferenceErrorで停止していました。
  const maskPaintCanvas=$('maskPaintCanvas');
  const maskPaintCtx=maskPaintCanvas.getContext('2d');
  let addBlurMask=null;
  let restoreSharpMask=null;
  let correctionMode='restore';
  let painting=false;
  let lastPaint=null;
  let paintRect=null;
  let paintScaleX=1;
  let paintScaleY=1;
  let paintRadius=20;
  let paintMinDistance=2;
  let queuedPaintPoint=null;
  let paintRafId=0;
  let toastTimer=0;

  const WORK_MAX=1600;
  const PREVIEW_MAX_SIDE=720;

  let previewSourceCache=null;
  let maskRevision=0;
  const effectiveMaskCache=new Map();
  const lensMaskCache=new Map();
  let depthMaskCacheKey='';
  let depthMaskCacheValue=null;
  let depthLayerCacheKey='';
  let depthLayerCacheValue=null;
  let renderScheduled=false;
  let pendingRenderMode='full';
  let settleRenderTimer=0;
  let manualOverlayBusy=false;
  let manualOverlayTitle='';
  let manualOverlaySub='';

  const FILTER_PRESETS={
    glow:{bright:110,sat:106,contrast:108,warmth:8,glow:24,subjectPop:16},
    bloom:{bright:114,sat:94,contrast:98,warmth:5,glow:14,subjectPop:10},
    clear:{bright:106,sat:107,contrast:116,warmth:2,glow:4,subjectPop:28},
    winter:{bright:110,sat:101,contrast:98,warmth:-12,glow:15,subjectPop:14},
    wooton:{bright:104,sat:116,contrast:106,warmth:22,glow:13,subjectPop:15},
    sakura:{bright:113,sat:94,contrast:90,warmth:10,glow:16,subjectPop:10}
  };
  const PRESETS=FILTER_PRESETS;

  function offscreen(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;return c;}
  function clamp01(v){return Math.max(0,Math.min(1,v));}
  function smoothstep(edge0,edge1,x){const t=clamp01((x-edge0)/(edge1-edge0));return t*t*(3-2*t);}
  function clamp255(v){return Math.max(0,Math.min(255,v));}
  function setStatus(msg){status.textContent=msg;}
  function setOverlayVisible(show,title='',sub=''){
    processingOverlay.hidden=!show;
    if(show){
      overlayText.textContent=title||'処理中…';
      overlaySub.textContent=sub||'しばらくお待ちください。';
    }
  }

  function updateOverlayFromState(){
    if(manualOverlayBusy){
      setOverlayVisible(true,manualOverlayTitle,manualOverlaySub);
      return;
    }
    if(segLoading){
      setOverlayVisible(true,'人物を判定中…','初回はモデル読み込みのため少し時間がかかることがあります。');
      return;
    }
    if(depthLoading){
      setOverlayVisible(true,'AI深度マップを生成中…','初回はDepth Anything V2のモデル読み込みに時間がかかることがあります。');
      return;
    }
    setOverlayVisible(false);
  }

  function updateSegUI(){
    segDot.className='dot';
    if(segLoading){segDot.classList.add('loading');segStatus.textContent='人物判定中…';}
    else if(segError){segDot.classList.add('error');segStatus.textContent='人物判定に失敗';}
    else if(autoBackgroundMask){segStatus.textContent='人物判定完了';}
    else {segDot.classList.add('loading');segStatus.textContent='人物判定の準備中';}
    updateOverlayFromState();
  }


  function updateDepthUI(message=''){
    depthDot.className='dot';
    if(depthLoading){
      depthDot.classList.add('loading');
      depthStatus.textContent=message||'AI深度マップを生成中…';
    }else if(depthError){
      depthDot.classList.add('error');
      depthStatus.textContent='AI深度マップに失敗（軽量方式で継続）';
    }else if(aiDepthCanvas){
      depthStatus.textContent=message||'AI深度マップ準備完了';
    }else{
      depthDot.classList.add('loading');
      depthStatus.textContent=message||'AI深度マップの準備中';
    }
    updateOverlayFromState();
  }

  function drawDepthPreview(){
    if(!aiDepthCanvas) return;
    const maxSide=620;
    const scale=Math.min(1,maxSide/Math.max(aiDepthCanvas.width,aiDepthCanvas.height));
    const w=Math.max(1,Math.round(aiDepthCanvas.width*scale));
    const h=Math.max(1,Math.round(aiDepthCanvas.height*scale));
    depthPreviewCanvas.width=w; depthPreviewCanvas.height=h;
    const dctx=depthPreviewCanvas.getContext('2d');
    dctx.imageSmoothingEnabled=true;
    if('imageSmoothingQuality' in dctx) dctx.imageSmoothingQuality='high';
    dctx.drawImage(aiDepthCanvas,0,0,aiDepthCanvas.width,aiDepthCanvas.height,0,0,w,h);
  }

  function invertDepthCanvas(canvas){
    if(!canvas) return;
    const cctx=canvas.getContext('2d');
    const img=cctx.getImageData(0,0,canvas.width,canvas.height), d=img.data;
    for(let i=0;i<d.length;i+=4){
      const v=255-d[i];
      d[i]=v;d[i+1]=v;d[i+2]=v;d[i+3]=255;
    }
    cctx.putImageData(img,0,0);
  }

  function sampleDepthForSubject(setSlider=true){
    if(!aiDepthCanvas || !autoBackgroundMask) return null;
    const maxSide=280;
    const scale=Math.min(1,maxSide/Math.max(workW,workH));
    const sw=Math.max(2,Math.round(workW*scale)), sh=Math.max(2,Math.round(workH*scale));
    const dc=offscreen(sw,sh), mc=offscreen(sw,sh);
    dc.getContext('2d').drawImage(aiDepthCanvas,0,0,aiDepthCanvas.width,aiDepthCanvas.height,0,0,sw,sh);
    mc.getContext('2d').drawImage(autoBackgroundMask,0,0,workW,workH,0,0,sw,sh);
    const dd=dc.getContext('2d').getImageData(0,0,sw,sh).data;
    const md=mc.getContext('2d').getImageData(0,0,sw,sh).data;
    const subject=[], background=[];
    for(let i=0;i<dd.length;i+=4){
      const bg=md[i+3]/255;
      if(bg<0.22) subject.push(dd[i]);
      else if(bg>0.86) background.push(dd[i]);
    }
    if(subject.length<20) return null;
    subject.sort((a,b)=>a-b);
    const s=subject[Math.floor(subject.length/2)];
    if(background.length>20){
      background.sort((a,b)=>a-b);
      const b=background[Math.floor(background.length/2)];
      // 深度マップは黒=手前、白=奥に統一する。
      // 被写体の中央値が背景より明らかに白い場合だけ自動反転。
      if(!depthManualInvert && !depthOrientationChecked && s>b+18){
        depthOrientationChecked=true;
        invertDepthCanvas(aiDepthCanvas);
        aiDepthRevision++;
        drawDepthPreview();
        return sampleDepthForSubject(setSlider);
      }
    }
    depthOrientationChecked=true;
    const pct=Math.max(0,Math.min(100,Math.round((s/255)*100)));
    if(setSlider){
      $('focusDepth').value=pct;
      $('focusDepthOut').textContent=pct;
      requestRender('full');
    }
    return pct;
  }

  function maybeAutoFocusDepth(){
    if(depthAutoFocused || !aiDepthCanvas || !autoBackgroundMask) return;
    const v=sampleDepthForSubject(true);
    if(v!==null) depthAutoFocused=true;
  }

  async function ensureDepthEstimatorLoaded(){
    if(depthEstimatorInstance) return depthEstimatorInstance;
    if(depthEstimatorPromise) return depthEstimatorPromise;
    depthEstimatorPromise=(async()=>{
      updateDepthUI('AI深度モデルを読み込み中…');
      const {pipeline}=await import(TRANSFORMERS_JS_URL);
      const pipe=await pipeline('depth-estimation',DEPTH_MODEL_ID,{
        device:'wasm',
        dtype:'q8',
        progress_callback:(info)=>{
          if(info && typeof info.progress==='number'){
            updateDepthUI(`AI深度モデルを読み込み中… ${Math.round(info.progress)}%`);
          }
        }
      });
      depthEstimatorInstance=pipe;
      return pipe;
    })();
    try{return await depthEstimatorPromise;}
    finally{depthEstimatorPromise=null;}
  }

  async function _computeAIDepthMapInternal(){
    if(!srcCanvas) return;
    depthLoading=true; depthError=null; aiDepthCanvas=null; depthAutoFocused=false; updateDepthUI();
    try{
      const estimator=await ensureDepthEstimatorLoaded();
      updateDepthUI('AI深度マップを解析中…');
      const inferMax=1280;
      const scale=Math.min(1,inferMax/Math.max(workW,workH));
      const iw=Math.max(2,Math.round(workW*scale)), ih=Math.max(2,Math.round(workH*scale));
      const input=offscreen(iw,ih), ictx=input.getContext('2d');
      ictx.imageSmoothingEnabled=true;
      if('imageSmoothingQuality' in ictx) ictx.imageSmoothingQuality='high';
      ictx.drawImage(srcCanvas,0,0,workW,workH,0,0,iw,ih);
      const blob=await new Promise((resolve,reject)=>input.toBlob(b=>b?resolve(b):reject(new Error('深度解析用画像を作成できませんでした')),'image/jpeg',0.92));
      const url=URL.createObjectURL(blob);
      let result;
      try{ result=await estimator(url); }
      finally{ URL.revokeObjectURL(url); }
      if(!result || !result.depth || !result.depth.data) throw new Error('深度マップを取得できませんでした');
      const raw=result.depth;
      const rc=offscreen(raw.width,raw.height), rctx=rc.getContext('2d');
      const img=rctx.createImageData(raw.width,raw.height), d=img.data;
      // Depth Anything V2 の相対深度は近い側が大きくなりやすいため、
      // まず反転して「黒=手前 / 白=奥」に統一する。人物マスク取得後に自動確認する。
      for(let i=0;i<raw.width*raw.height;i++){
        const srcV=raw.data[i];
        const v=255-srcV;
        const p=i*4; d[p]=v;d[p+1]=v;d[p+2]=v;d[p+3]=255;
      }
      rctx.putImageData(img,0,0);
      aiDepthCanvas=offscreen(workW,workH);
      const actx=aiDepthCanvas.getContext('2d');
      actx.imageSmoothingEnabled=true;
      if('imageSmoothingQuality' in actx) actx.imageSmoothingQuality='high';
      actx.filter='blur(1.15px)';
      actx.drawImage(rc,0,0,raw.width,raw.height,0,0,workW,workH);
      actx.filter='none';
      aiDepthRevision++;
      drawDepthPreview();
      updateDepthUI('AI深度マップ準備完了');
      maybeAutoFocusDepth();
      requestRender('full');
    }catch(err){
      console.error(err); depthError=String(err?.message||err); aiDepthCanvas=null; updateDepthUI(); requestRender('full');
    }finally{
      depthLoading=false; updateDepthUI(aiDepthCanvas?'AI深度マップ準備完了':'');
    }
  }

  function computeAIDepthMap(){
    if(depthComputePromise) return depthComputePromise;
    depthComputePromise=_computeAIDepthMapInternal().finally(()=>{depthComputePromise=null;});
    return depthComputePromise;
  }

  function invalidatePreviewCache(){
    previewSourceCache=null;
  }

  function bumpMaskRevision(){
    maskRevision++;
    effectiveMaskCache.clear();
    lensMaskCache.clear();
    depthMaskCacheKey='';
    depthMaskCacheValue=null;
    depthLayerCacheKey='';
    depthLayerCacheValue=null;
  }

  function getPreviewSize(){
    const maxSide=Math.max(workW,workH);
    if(!maxSide || maxSide<=PREVIEW_MAX_SIDE) return {w:workW,h:workH,scale:1};
    const scale=PREVIEW_MAX_SIDE/maxSide;
    return {w:Math.max(1,Math.round(workW*scale)),h:Math.max(1,Math.round(workH*scale)),scale};
  }

  function getPreviewSource(){
    const size=getPreviewSize();
    if(size.scale===1) return {canvas:srcCanvas,w:workW,h:workH};
    if(!previewSourceCache || previewSourceCache.w!==size.w || previewSourceCache.h!==size.h){
      const c=offscreen(size.w,size.h);
      const cctx=c.getContext('2d');
      cctx.imageSmoothingEnabled=true;
      if('imageSmoothingQuality' in cctx) cctx.imageSmoothingQuality='medium';
      cctx.drawImage(srcCanvas,0,0,workW,workH,0,0,size.w,size.h);
      previewSourceCache={canvas:c,w:size.w,h:size.h};
    }
    return previewSourceCache;
  }

  async function decodeFile(file){
    if('createImageBitmap' in window){try{return await createImageBitmap(file);}catch(e){}}
    return await new Promise((resolve,reject)=>{
      const reader=new FileReader();
      reader.onerror=()=>reject(new Error('FileReaderで読み込めませんでした'));
      reader.onload=()=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('画像デコードに失敗しました'));img.src=reader.result;};
      reader.readAsDataURL(file);
    });
  }

  async function ensureImageSegmenterLoaded(){
    if(imageSegmenterInstance) return imageSegmenterInstance;
    const vision = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`);
    const filesetResolver = await vision.FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`);
    imageSegmenterInstance = await vision.ImageSegmenter.createFromOptions(filesetResolver,{
      baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite'},
      runningMode:'IMAGE', outputCategoryMask:false, outputConfidenceMasks:true
    });
    return imageSegmenterInstance;
  }

  async function computeAutoBackgroundMask(){
    if(!fullImg || segLoading) return;
    segLoading=true;segError=null;updateSegUI();
    try{
      const segmenter=await ensureImageSegmenterLoaded();
      const result=segmenter.segment(fullImg);
      const confidenceMask=result.confidenceMasks[0];
      const floats=confidenceMask.getAsFloat32Array();
      const mw=confidenceMask.width, mh=confidenceMask.height;

      // と同じ考え方：人物信頼度をそのまま反転して背景マスクにする
      const raw=offscreen(mw,mh), rctx=raw.getContext('2d');
      const imgData=rctx.createImageData(mw,mh);
      for(let i=0;i<floats.length;i++){
        const bgAlpha=Math.round((1-floats[i])*255);
        imgData.data[i*4]=255;
        imgData.data[i*4+1]=255;
        imgData.data[i*4+2]=255;
        imgData.data[i*4+3]=bgAlpha;
      }
      rctx.putImageData(imgData,0,0);
      confidenceMask.close();
      if(typeof result.close==='function') result.close();

      autoBackgroundMask=offscreen(workW,workH);
      const mctx=autoBackgroundMask.getContext('2d');
      mctx.imageSmoothingEnabled=true;
      if('imageSmoothingQuality' in mctx) mctx.imageSmoothingQuality='high';
      mctx.drawImage(raw,0,0,workW,workH);

      addBlurMask=offscreen(workW,workH);
      restoreSharpMask=offscreen(workW,workH);
      maskPaintCanvas.width=workW;maskPaintCanvas.height=workH;
      maskPaintCtx.clearRect(0,0,workW,workH);
      bumpMaskRevision();
    }catch(err){
      console.error(err);segError=String(err?.message||err);autoBackgroundMask=null;
    }finally{segLoading=false;updateSegUI();maybeAutoFocusDepth();render();}
  }

  function remapBackgroundMaskAlpha(maskCanvas,w,h,low=0.34,high=0.78){
    const out=offscreen(w,h), octx=out.getContext('2d');
    octx.drawImage(maskCanvas,0,0,w,h);
    const img=octx.getImageData(0,0,w,h), d=img.data;
    for(let i=0;i<d.length;i+=4){
      const a=d[i+3]/255;
      let na=smoothstep(low,high,a);
      if(na<0.015) na=0;
      d[i]=255; d[i+1]=255; d[i+2]=255; d[i+3]=Math.round(na*255);
    }
    octx.putImageData(img,0,0);
    return out;
  }

  function softenMaskEdge(maskCanvas,w,h,blurPx){
    if(blurPx<=0) return maskCanvas;
    const out=offscreen(w,h), octx=out.getContext('2d');
    octx.filter=`blur(${blurPx}px)`;
    octx.drawImage(maskCanvas,0,0,w,h);
    octx.filter='none';
    return out;
  }

  function expandAlphaMask(maskCanvas,w,h,radiusPx){
    const r=Math.max(0,Math.round(radiusPx));
    if(r<=0) return maskCanvas;
    const out=offscreen(w,h), octx=out.getContext('2d');
    for(let oy=-r;oy<=r;oy++){
      for(let ox=-r;ox<=r;ox++){
        if((ox*ox)+(oy*oy)>(r*r)) continue;
        octx.drawImage(maskCanvas,ox,oy,w,h);
      }
    }
    return out;
  }

  function buildProtectedBackgroundMask(baseMask,w,h){
    if(!baseMask) return null;

    const edgeSoft=Math.max(0,Math.min(1,+$('edgeSoftness').value/100));

    // 背景ぼかし量は保ちつつ、人物境界だけをなだらかにする。
    let refined=remapBackgroundMaskAlpha(baseMask,w,h,0.34,0.80);

    // 人物側へ少し保護帯を広げる。髪・耳・装飾へのボケ侵入を抑える。
    const subjectMask=subjectMaskFromBackgroundMask(refined,w,h);
    const protectGrow=Math.max(1,Math.round(Math.max(w,h)*(0.0015 + edgeSoft*0.0028)));
    const expandedSubject=expandAlphaMask(subjectMask,w,h,protectGrow);
    const protectBlur=Math.max(2,Math.round(Math.max(w,h)*(0.0025 + edgeSoft*0.0042)));
    const protect=offscreen(w,h), pctx=protect.getContext('2d');
    pctx.filter=`blur(${protectBlur}px)`;
    pctx.drawImage(expandedSubject,0,0,w,h);
    pctx.filter='none';
    const protectImg=pctx.getImageData(0,0,w,h), pd=protectImg.data;
    const high=0.58-edgeSoft*0.16;
    for(let i=0;i<pd.length;i+=4){
      const a=pd[i+3]/255;
      const na=smoothstep(0.08,high,a);
      pd[i]=255; pd[i+1]=255; pd[i+2]=255; pd[i+3]=Math.round(na*255);
    }
    pctx.putImageData(protectImg,0,0);

    const cut=offscreen(w,h), cctx=cut.getContext('2d');
    cctx.drawImage(refined,0,0,w,h);
    cctx.globalCompositeOperation='destination-out';
    cctx.drawImage(protect,0,0,w,h);
    cctx.globalCompositeOperation='source-over';

    refined=softenMaskEdge(cut,w,h,1.2+edgeSoft*2.8);
    refined=remapBackgroundMaskAlpha(refined,w,h,0.05,0.985);
    return refined;
  }

  function buildEffectiveBackgroundMask(w,h){
    if(!autoBackgroundMask && !addBlurMask) return null;
    const cacheKey=`${maskRevision}:${w}x${h}:edge${Math.round(+$('edgeSoftness').value)}`;
    if(effectiveMaskCache.has(cacheKey)) return effectiveMaskCache.get(cacheKey);
    const out=offscreen(w,h), octx=out.getContext('2d');
    if(autoBackgroundMask) octx.drawImage(autoBackgroundMask,0,0,workW,workH,0,0,w,h);
    if(addBlurMask) octx.drawImage(addBlurMask,0,0,workW,workH,0,0,w,h);
    if(restoreSharpMask){
      octx.globalCompositeOperation='destination-out';
      octx.drawImage(restoreSharpMask,0,0,workW,workH,0,0,w,h);
      octx.globalCompositeOperation='source-over';
    }
    const refined=buildProtectedBackgroundMask(out,w,h);
    effectiveMaskCache.set(cacheKey,refined);
    return refined;
  }


  // AI深度DOF用の背景マスク。
  // 従来の保護帯より広げ過ぎず、髪・耳の縁だけを柔らかく保護する。
  function buildLensBackgroundMask(w,h){
    if(!autoBackgroundMask && !addBlurMask) return null;
    const edge=Math.max(0,Math.min(1,+$('edgeSoftness').value/100));
    const cacheKey=`${maskRevision}:${w}x${h}:lens:${Math.round(edge*100)}`;
    if(lensMaskCache.has(cacheKey)) return lensMaskCache.get(cacheKey);
    const base=offscreen(w,h), bctx=base.getContext('2d');
    if(autoBackgroundMask) bctx.drawImage(autoBackgroundMask,0,0,workW,workH,0,0,w,h);
    if(addBlurMask) bctx.drawImage(addBlurMask,0,0,workW,workH,0,0,w,h);
    if(restoreSharpMask){
      bctx.globalCompositeOperation='destination-out';
      bctx.drawImage(restoreSharpMask,0,0,workW,workH,0,0,w,h);
      bctx.globalCompositeOperation='source-over';
    }
    let refined=remapBackgroundMaskAlpha(base,w,h,0.30,0.82);
    const subject=subjectMaskFromBackgroundMask(refined,w,h);
    const grow=Math.max(1,Math.round(Math.max(w,h)*(0.0007+edge*0.0015)));
    const expanded=expandAlphaMask(subject,w,h,grow);
    const protect=offscreen(w,h), pctx=protect.getContext('2d');
    const feather=Math.max(1.2,Math.max(w,h)*(0.0012+edge*0.0023));
    pctx.filter=`blur(${feather}px)`;
    pctx.drawImage(expanded,0,0,w,h);
    pctx.filter='none';
    const cut=offscreen(w,h), cctx=cut.getContext('2d');
    cctx.drawImage(refined,0,0,w,h);
    cctx.globalCompositeOperation='destination-out';
    cctx.globalAlpha=0.72;
    cctx.drawImage(protect,0,0,w,h);
    cctx.globalAlpha=1;
    cctx.globalCompositeOperation='source-over';
    refined=softenMaskEdge(cut,w,h,1.1+edge*2.6);
    refined=remapBackgroundMaskAlpha(refined,w,h,0.035,0.97);
    lensMaskCache.set(cacheKey,refined);
    return refined;
  }


  // と同じ考え方: 一度小さく縮めてから拡大して滑らかなぼかしを作る
  function buildScaleBlur(source,w,h,radiusPx){
    if(radiusPx<=0){const c=offscreen(w,h);c.getContext('2d').drawImage(source,0,0,w,h);return c;}
    const targetScale=Math.min(1,1/(radiusPx*.55));
    const targetW=Math.max(2,Math.round(w*targetScale));
    const targetH=Math.max(2,Math.round(h*targetScale));
    let cw=w,ch=h;
    let cur=offscreen(cw,ch);
    let cctx=cur.getContext('2d'); cctx.imageSmoothingEnabled=true; if('imageSmoothingQuality' in cctx)cctx.imageSmoothingQuality='high'; cctx.drawImage(source,0,0,w,h,0,0,cw,ch);
    while(cw>targetW && ch>targetH && cw>4 && ch>4){
      const nw=Math.max(2,Math.round(cw/2)), nh=Math.max(2,Math.round(ch/2));
      const next=offscreen(nw,nh), nctx=next.getContext('2d'); nctx.imageSmoothingEnabled=true; if('imageSmoothingQuality' in nctx)nctx.imageSmoothingQuality='high'; nctx.drawImage(cur,0,0,cw,ch,0,0,nw,nh); cur=next; cw=nw; ch=nh;
    }
    if(cw!==targetW||ch!==targetH){const trimmed=offscreen(targetW,targetH), tctx=trimmed.getContext('2d');tctx.imageSmoothingEnabled=true;if('imageSmoothingQuality' in tctx)tctx.imageSmoothingQuality='high';tctx.drawImage(cur,0,0,cw,ch,0,0,targetW,targetH);cur=trimmed;cw=targetW;ch=targetH;}
    const out=offscreen(w,h), octx=out.getContext('2d');octx.imageSmoothingEnabled=true;if('imageSmoothingQuality' in octx)octx.imageSmoothingQuality='high';octx.drawImage(cur,0,0,cw,ch,0,0,w,h);return out;
  }

  function subjectMaskFromBackgroundMask(mask,w,h){
    const out=offscreen(w,h);
    const octx=out.getContext('2d');
    octx.fillStyle='#fff';
    octx.fillRect(0,0,w,h);
    octx.globalCompositeOperation='destination-out';
    octx.drawImage(mask,0,0,w,h);
    octx.globalCompositeOperation='source-over';
    return out;
  }


  function apertureStrength(fNumber){
    const f=Math.max(1.4,Math.min(16,+fNumber||2.8));
    const inv=1/f, invMin=1/16, invMax=1/1.4;
    return clamp01((inv-invMin)/(invMax-invMin));
  }

  // 人物マスクから「背景の距離らしさ」を作る軽量な距離場。
  // 単純な輪郭拡張ではなく、人物からの距離 + 画面上の遠近方向を混ぜることで、
  // 人物の周囲だけが島状に残る見え方を抑える。
  function buildProgressiveDepthLayers(backgroundMask,w,h,depth){
    const d=clamp01(depth);
    if(!backgroundMask || d<=0.001) return null;

    const depthStep=Math.round(d*100);
    const cacheKey=`${maskRevision}:${w}x${h}:${depthStep}:edge${Math.round(+$('edgeSoftness').value)}`;
    if(depthLayerCacheKey===cacheKey && depthLayerCacheValue) return depthLayerCacheValue;

    const maxSide=500;
    const scale=Math.min(1,maxSide/Math.max(w,h));
    const sw=Math.max(2,Math.round(w*scale));
    const sh=Math.max(2,Math.round(h*scale));

    const bgSmall=offscreen(sw,sh), bgctx=bgSmall.getContext('2d');
    bgctx.imageSmoothingEnabled=true;
    if('imageSmoothingQuality' in bgctx) bgctx.imageSmoothingQuality='medium';
    bgctx.drawImage(backgroundMask,0,0,w,h,0,0,sw,sh);
    const bgImg=bgctx.getImageData(0,0,sw,sh), bg=bgImg.data;

    const n=sw*sh;
    const INF=1e6;
    const dist=new Float32Array(n);
    let minY=sh, maxY=0, subjectCount=0;

    for(let y=0;y<sh;y++){
      for(let x=0;x<sw;x++){
        const i=y*sw+x;
        const ba=bg[i*4+3]/255;
        const isSubject=ba<0.20;
        dist[i]=isSubject?0:INF;
        if(isSubject){subjectCount++; if(y<minY)minY=y; if(y>maxY)maxY=y;}
      }
    }

    if(!subjectCount){
      depthLayerCacheKey=cacheKey;
      depthLayerCacheValue=null;
      return null;
    }

    const diag=1.41421356;
    for(let y=0;y<sh;y++){
      for(let x=0;x<sw;x++){
        const i=y*sw+x;
        let v=dist[i];
        if(x>0) v=Math.min(v,dist[i-1]+1);
        if(y>0) v=Math.min(v,dist[i-sw]+1);
        if(x>0&&y>0) v=Math.min(v,dist[i-sw-1]+diag);
        if(x<sw-1&&y>0) v=Math.min(v,dist[i-sw+1]+diag);
        dist[i]=v;
      }
    }
    for(let y=sh-1;y>=0;y--){
      for(let x=sw-1;x>=0;x--){
        const i=y*sw+x;
        let v=dist[i];
        if(x<sw-1) v=Math.min(v,dist[i+1]+1);
        if(y<sh-1) v=Math.min(v,dist[i+sw]+1);
        if(x<sw-1&&y<sh-1) v=Math.min(v,dist[i+sw+1]+diag);
        if(x>0&&y<sh-1) v=Math.min(v,dist[i+sw-1]+diag);
        dist[i]=v;
      }
    }

    const focusBottom=Math.max(1,maxY);
    const maxDim=Math.max(sw,sh);
    const nearDist=maxDim*(0.018 + d*0.055);
    const farDist=nearDist + maxDim*(0.12 + (1-d)*0.075);

    // 累積マスク。後ろのレイヤーほど遠い背景だけに効く。
    const thresholds=[[0.03,0.26],[0.20,0.48],[0.43,0.72],[0.67,0.96]];
    const layers=thresholds.map(()=>offscreen(sw,sh));
    const layerData=layers.map(c=>c.getContext('2d').createImageData(sw,sh));

    for(let y=0;y<sh;y++){
      const farByY=clamp01((focusBottom-y)/(sh*0.60));
      const foregroundByY=clamp01((y-focusBottom)/(sh*0.32));
      for(let x=0;x<sw;x++){
        const i=y*sw+x, p=i*4;
        const ba=bg[p+3]/255;
        if(ba<=0.002) continue;

        const radial=smoothstep(nearDist,farDist,dist[i]);
        // 背景の上側は遠景になりやすい。下側は手前として少しボケを抑える。
        let weight=radial*(0.48+0.52*farByY) + farByY*0.30;
        if(y>focusBottom) weight*=0.55+0.25*foregroundByY;
        weight=clamp01(weight*d + radial*(1-d)*0.55);

        for(let k=0;k<layers.length;k++){
          const [a,b]=thresholds[k];
          const alpha=ba*smoothstep(a,b,weight);
          const ld=layerData[k].data;
          ld[p]=255; ld[p+1]=255; ld[p+2]=255; ld[p+3]=Math.round(alpha*255);
        }
      }
    }

    for(let k=0;k<layers.length;k++) layers[k].getContext('2d').putImageData(layerData[k],0,0);

    depthLayerCacheKey=cacheKey;
    depthLayerCacheValue={layers,sw,sh};
    return depthLayerCacheValue;
  }


  let dofGLState=null;
  function getDofGLState(){
    if(dofGLState) return dofGLState;
    const glCanvas=document.createElement('canvas');
    const gl=glCanvas.getContext('webgl2',{alpha:true,premultipliedAlpha:false,antialias:false,preserveDrawingBuffer:true});
    if(!gl) return null;
    const vs=`#version 300 es
precision highp float;
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
void main(){v_uv=a_uv;gl_Position=vec4(a_pos,0.0,1.0);}`;
    const fs=`#version 300 es
precision highp float;
uniform sampler2D u_image;
uniform sampler2D u_depth;
uniform sampler2D u_mask;
uniform vec2 u_texel;
uniform float u_focus;
uniform float u_dof;
uniform float u_aperture;
uniform float u_maxRadius;
in vec2 v_uv;
out vec4 outColor;
const int N=24;
const vec2 OFF[N]=vec2[N](
 vec2(0.000,0.000), vec2(0.527,0.085), vec2(-0.040,0.536), vec2(-0.550,0.025),
 vec2(-0.260,-0.520), vec2(0.410,-0.450), vec2(0.820,0.300), vec2(0.250,0.850),
 vec2(-0.650,0.650), vec2(-0.900,-0.220), vec2(-0.350,-0.880), vec2(0.580,-0.760),
 vec2(0.980,-0.080), vec2(0.760,0.640), vec2(0.080,0.990), vec2(-0.720,0.820),
 vec2(-0.990,0.180), vec2(-0.760,-0.620), vec2(-0.080,-0.990), vec2(0.720,-0.820),
 vec2(0.360,0.250), vec2(-0.310,0.310), vec2(-0.300,-0.300), vec2(0.300,-0.320));
void main(){
 vec4 src=texture(u_image,v_uv);
 float mask=texture(u_mask,v_uv).a;
 if(mask<0.015){outColor=src;return;}
 float z=texture(u_depth,v_uv).r;
 float focusBand=mix(0.010,0.155,u_dof);
 float delta=abs(z-u_focus);
 float coc=smoothstep(focusBand,focusBand+0.30,delta);
 if(z<u_focus) coc*=0.78;
 float radius=u_maxRadius*u_aperture*coc;
 if(radius<0.30){outColor=src;return;}
 vec3 sum=src.rgb;
 float total=1.0;
 for(int i=1;i<N;i++){
   vec2 suv=clamp(v_uv+OFF[i]*u_texel*radius,vec2(0.001),vec2(0.999));
   float sm=texture(u_mask,suv).a;
   if(sm<0.08) continue;
   float sz=texture(u_depth,suv).r;
   float depthCompat=1.0-smoothstep(0.07,0.34,abs(sz-z));
   float w=smoothstep(0.08,0.72,sm)*(0.32+0.68*depthCompat);
   sum+=texture(u_image,suv).rgb*w;
   total+=w;
 }
 vec3 blurred=sum/max(total,0.001);
 float edge=smoothstep(0.035,0.92,mask);
 float mixAmt=coc*edge;
 outColor=vec4(mix(src.rgb,blurred,mixAmt),src.a);
}`;
    function compile(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s)||'shader compile failed');return s;}
    try{
      const program=gl.createProgram();
      gl.attachShader(program,compile(gl.VERTEX_SHADER,vs));
      gl.attachShader(program,compile(gl.FRAGMENT_SHADER,fs));
      gl.linkProgram(program);
      if(!gl.getProgramParameter(program,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program)||'shader link failed');
      const verts=new Float32Array([
        -1,-1, 0,1,   1,-1, 1,1,  -1,1, 0,0,
        -1,1, 0,0,    1,-1, 1,1,   1,1, 1,0
      ]);
      const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,verts,gl.STATIC_DRAW);
      const stride=4*4;
      const aPos=gl.getAttribLocation(program,'a_pos'), aUv=gl.getAttribLocation(program,'a_uv');
      gl.enableVertexAttribArray(aPos);gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,stride,0);
      gl.enableVertexAttribArray(aUv);gl.vertexAttribPointer(aUv,2,gl.FLOAT,false,stride,2*4);
      const tex=[gl.createTexture(),gl.createTexture(),gl.createTexture()];
      dofGLState={glCanvas,gl,program,buffer,tex};
      return dofGLState;
    }catch(err){console.error('WebGL DOF init failed',err);return null;}
  }

  function uploadDofTexture(gl,tex,unit,source){
    gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,tex);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,source);
  }

  function applyDepthAwareDofWebGL(source,depthCanvas,backgroundMask,w,h,v){
    if(!source||!depthCanvas||!backgroundMask) return null;
    const state=getDofGLState(); if(!state) return null;
    const {glCanvas,gl,program,tex}=state;
    glCanvas.width=w;glCanvas.height=h;gl.viewport(0,0,w,h);gl.useProgram(program);
    uploadDofTexture(gl,tex[0],0,source);
    uploadDofTexture(gl,tex[1],1,depthCanvas);
    uploadDofTexture(gl,tex[2],2,backgroundMask);
    gl.uniform1i(gl.getUniformLocation(program,'u_image'),0);
    gl.uniform1i(gl.getUniformLocation(program,'u_depth'),1);
    gl.uniform1i(gl.getUniformLocation(program,'u_mask'),2);
    gl.uniform2f(gl.getUniformLocation(program,'u_texel'),1/w,1/h);
    gl.uniform1f(gl.getUniformLocation(program,'u_focus'),Math.max(0,Math.min(1,v.focus)));
    gl.uniform1f(gl.getUniformLocation(program,'u_dof'),Math.max(0,Math.min(1,v.depth)));
    const aperture=0.12+apertureStrength(v.aperture)*1.18;
    gl.uniform1f(gl.getUniformLocation(program,'u_aperture'),aperture);
    gl.uniform1f(gl.getUniformLocation(program,'u_maxRadius'),Math.max(1,v.blur));
    gl.disable(gl.BLEND);gl.drawArrays(gl.TRIANGLES,0,6);
    const out=offscreen(w,h), octx=out.getContext('2d');
    octx.drawImage(glCanvas,0,0,w,h);
    return out;
  }


  function drawShapePath(c,shape,r){
    if(shape==='hexagon'){
      for(let i=0;i<6;i++){
        const a=(-Math.PI/2)+(Math.PI*2*i/6);
        const x=Math.cos(a)*r, y=Math.sin(a)*r;
        if(i===0) c.moveTo(x,y); else c.lineTo(x,y);
      }
      c.closePath();
      return;
    }
    if(shape==='star'){
      const inner=r*0.46;
      for(let i=0;i<10;i++){
        const rr=(i%2===0)?r:inner;
        const a=(-Math.PI/2)+(Math.PI*i/5);
        const x=Math.cos(a)*rr, y=Math.sin(a)*rr;
        if(i===0) c.moveTo(x,y); else c.lineTo(x,y);
      }
      c.closePath();
      return;
    }
    c.arc(0,0,r,0,Math.PI*2);
  }

  function getPixelLuma(data,idx){
    return (data[idx]*0.2126 + data[idx+1]*0.7152 + data[idx+2]*0.0722)/255;
  }

  function drawBokehElement(ctx,shape,c,radius,alpha,fringePx){
    const preserve=0.94;
    const cr=Math.round(c.r*preserve + Math.min(255,c.r+8)*(1-preserve));
    const cg=Math.round(c.g*preserve + Math.min(255,c.g+8)*(1-preserve));
    const cb=Math.round(c.b*preserve + Math.min(255,c.b+8)*(1-preserve));

    ctx.save();
    ctx.translate(c.x,c.y);
    if(shape==='hexagon') ctx.rotate((c.x*0.013 + c.y*0.009)%Math.PI);
    else if(shape==='star') ctx.rotate((c.x*0.017 + c.y*0.011)%Math.PI);

    ctx.beginPath();
    drawShapePath(ctx,shape,radius);
    const grad=ctx.createRadialGradient(0,0,0,0,0,radius);
    if(shape==='circle'){
      grad.addColorStop(0,`rgba(${cr},${cg},${cb},${Math.min(0.55,alpha*0.72)})`);
      grad.addColorStop(0.38,`rgba(${cr},${cg},${cb},${alpha*0.42})`);
      grad.addColorStop(0.74,`rgba(${cr},${cg},${cb},${alpha*0.17})`);
      grad.addColorStop(1,`rgba(${cr},${cg},${cb},0)`);
    }else{
      grad.addColorStop(0,`rgba(${cr},${cg},${cb},${Math.min(0.70,alpha*0.84)})`);
      grad.addColorStop(0.44,`rgba(${cr},${cg},${cb},${alpha*0.46})`);
      grad.addColorStop(0.82,`rgba(${cr},${cg},${cb},${alpha*0.13})`);
      grad.addColorStop(1,`rgba(${cr},${cg},${cb},0)`);
    }
    ctx.fillStyle=grad;
    ctx.fill();

    // Keep the source color harmony; only add a very subtle core so highlights do not pop unnaturally.
    const coreAlpha=(shape==='circle' ? alpha*0.14 : alpha*0.22);
    if(coreAlpha>0.01){
      ctx.beginPath();
      drawShapePath(ctx,shape,radius*(shape==='circle' ? 0.24 : 0.30));
      const core=ctx.createRadialGradient(0,0,0,0,0,radius*(shape==='circle' ? 0.24 : 0.30));
      core.addColorStop(0,`rgba(${Math.min(255,cr+10)},${Math.min(255,cg+10)},${Math.min(255,cb+10)},${coreAlpha})`);
      core.addColorStop(1,'rgba(255,255,255,0)');
      ctx.fillStyle=core;
      ctx.fill();
    }

    // Subtle chromatic edge tint to mimic lens aberration without floating colors.
    if(fringePx>0.04){
      const lineW=Math.max(0.38, radius*0.045 + fringePx*0.40);
      const edgeAlpha=Math.min(0.12, alpha*0.09 + fringePx*0.012);
      const offsets=[
        {dx: fringePx*0.55, dy:-fringePx*0.20, color:`rgba(${Math.min(255,cr+28)},${Math.max(0,cg-6)},${Math.max(0,cb-10)},${edgeAlpha})`},
        {dx:-fringePx*0.52, dy: fringePx*0.22, color:`rgba(${Math.max(0,cr-10)},${Math.max(0,cg-4)},${Math.min(255,cb+30)},${edgeAlpha*0.92})`}
      ];
      ctx.globalCompositeOperation='screen';
      for(const o of offsets){
        ctx.save();
        ctx.translate(o.dx,o.dy);
        ctx.beginPath();
        drawShapePath(ctx,shape,Math.max(0.8, radius - lineW*0.12));
        ctx.strokeStyle=o.color;
        ctx.lineWidth=lineW;
        ctx.stroke();
        ctx.restore();
      }
      ctx.globalCompositeOperation='source-over';
    }
    ctx.restore();
  }

  function applyBokehShapes(base,source,depthCanvas,backgroundMask,w,h,v,interactive=false){
    if(!base || !source || !depthCanvas || !backgroundMask) return base;
    if(v.bokehShape==='off' || v.bokehSize<=0 || v.bokehIntensity<=0.001) return base;

    const sctx=source.getContext('2d',{willReadFrequently:true});
    const dctx=depthCanvas.getContext('2d',{willReadFrequently:true});
    const mctx=backgroundMask.getContext('2d',{willReadFrequently:true});
    const srcData=sctx.getImageData(0,0,w,h).data;
    const depthData=dctx.getImageData(0,0,w,h).data;
    const maskData=mctx.getImageData(0,0,w,h).data;

    const density=Math.max(0,Math.min(1,v.bokehDensity));
    const fringe=Math.max(0,Math.min(1,v.bokehFringe));
    const threshold=Math.max(0.44, 0.68 - v.bokehIntensity*0.15 - density*0.08);
    const focusBand=0.010 + (0.155-0.010)*Math.max(0,Math.min(1,v.depth));
    const aperture=0.10 + apertureStrength(v.aperture)*0.95;
    const stepBase=6.5 - density*4.2;
    const step=Math.max(1, Math.round(interactive ? Math.max(2,stepBase+1.0) : stepBase));
    const radiusBase=Math.max(1.4, 0.9 + (v.bokehSize*0.07) + (v.blur*0.10) + aperture*1.05);
    const maxCandidates=interactive
      ? Math.min(420, Math.max(140, Math.round((w*h)/(step*step*140))))
      : Math.min(4200, Math.max(500, Math.round((w*h)/(step*step*10))));
    const candidates=[];

    for(let y=2; y<h-2; y+=step){
      for(let x=2; x<w-2; x+=step){
        const idx=(y*w+x)*4;
        const maskA=maskData[idx+3]/255;
        if(maskA<0.12) continue;
        const z=depthData[idx]/255;
        const delta=z-v.focus;
        if(delta<=focusBand*0.15) continue;
        const coc=smoothstep(focusBand, focusBand+0.28, Math.abs(z-v.focus));
        if(coc<0.10) continue;
        const lum=getPixelLuma(srcData,idx);
        if(lum<threshold) continue;

        const left=getPixelLuma(srcData,idx-4), right=getPixelLuma(srcData,idx+4), up=getPixelLuma(srcData,idx-w*4), down=getPixelLuma(srcData,idx+w*4);
        const localAvg=(left+right+up+down)*0.25;
        const prominence=lum-localAvg;
        if(prominence<0.003 && lum<0.78) continue;

        const farFactor=smoothstep(v.focus+focusBand*0.28,1.0,z);
        const strength=((lum-threshold)/Math.max(0.0001,1-threshold)) * (0.34+0.66*coc) * (0.20+0.80*farFactor) * (0.36+0.64*maskA);
        if(strength<=0.010) continue;

        candidates.push({
          x,y,z,coc,lum,
          r:srcData[idx],g:srcData[idx+1],b:srcData[idx+2],
          score:strength + Math.max(0,prominence)*0.22
        });
      }
    }

    if(!candidates.length) return base;
    candidates.sort((a,b)=>b.score-a.score);
    const selected=candidates.slice(0,maxCandidates);

    const overlay=offscreen(w,h);
    const octx=overlay.getContext('2d');
    octx.imageSmoothingEnabled=true;
    if('imageSmoothingQuality' in octx) octx.imageSmoothingQuality='high';
    octx.globalCompositeOperation='source-over';

    const fringePxBase=fringe * (0.15 + radiusBase*0.05 + v.bokehSize*0.008);

    for(const c of selected){
      const radius=radiusBase*(0.82 + c.coc*0.56) * (0.94 + c.score*0.16);
      const alphaBase=(shape => shape==='circle' ? 0.18 : 0.28)(v.bokehShape);
      const alpha=Math.min(0.34, (0.03 + v.bokehIntensity*alphaBase) * (0.34 + c.score*0.62));
      const fringePx=fringePxBase * (0.45 + c.coc*0.55);
      drawBokehElement(octx,v.bokehShape,c,radius,alpha,fringePx);
    }

    const softened=offscreen(w,h);
    const sfctx=softened.getContext('2d');
    const blurPx=Math.max(0.22, radiusBase*(0.035 + fringe*0.018));
    if('filter' in sfctx) sfctx.filter=`blur(${blurPx.toFixed(2)}px)`;
    sfctx.globalCompositeOperation='source-over';
    sfctx.drawImage(overlay,0,0);
    if('filter' in sfctx) sfctx.filter='none';
    sfctx.globalCompositeOperation='destination-in';
    sfctx.drawImage(backgroundMask,0,0,w,h);
    sfctx.globalCompositeOperation='source-over';

    const out=offscreen(w,h);
    const outCtx=out.getContext('2d');
    outCtx.drawImage(base,0,0);

    // Blend in gently to keep the original background hue/saturation intact.
    outCtx.globalCompositeOperation='soft-light';
    outCtx.globalAlpha=Math.min(0.58, 0.10 + v.bokehIntensity*0.30);
    outCtx.drawImage(softened,0,0);

    outCtx.globalCompositeOperation='source-over';
    outCtx.globalAlpha=Math.min(0.26, 0.04 + v.bokehIntensity*0.16);
    outCtx.drawImage(softened,0,0);

    outCtx.globalAlpha=1;
    outCtx.globalCompositeOperation='source-over';
    return out;
  }

  function sharpenCanvas(source,w,h,amount){
    const srcC=offscreen(w,h);
    const sctx=srcC.getContext('2d');
    sctx.drawImage(source,0,0,w,h);
    const src=sctx.getImageData(0,0,w,h);
    const s=src.data;
    const out=offscreen(w,h);
    const octx=out.getContext('2d');
    const dst=octx.createImageData(w,h);
    const d=dst.data;
    const t=Math.max(0,Math.min(1,amount/40));
    const center=1+4*t;
    const edge=-t;
    const clamp255=v=>Math.max(0,Math.min(255,v));

    for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
        const idx=(y*w+x)*4;
        if(x===0||y===0||x===w-1||y===h-1){
          d[idx]=s[idx]; d[idx+1]=s[idx+1]; d[idx+2]=s[idx+2]; d[idx+3]=s[idx+3];
          continue;
        }
        const top=((y-1)*w+x)*4;
        const bottom=((y+1)*w+x)*4;
        const left=(y*w+(x-1))*4;
        const right=(y*w+(x+1))*4;
        d[idx]=clamp255(s[idx]*center+s[top]*edge+s[bottom]*edge+s[left]*edge+s[right]*edge);
        d[idx+1]=clamp255(s[idx+1]*center+s[top+1]*edge+s[bottom+1]*edge+s[left+1]*edge+s[right+1]*edge);
        d[idx+2]=clamp255(s[idx+2]*center+s[top+2]*edge+s[bottom+2]*edge+s[left+2]*edge+s[right+2]*edge);
        d[idx+3]=s[idx+3];
      }
    }
    octx.putImageData(dst,0,0);
    return out;
  }

  function applySubjectPop(source,w,h,backgroundMask,amount){
    if(!backgroundMask || amount<=0) return source;
    const subjectMask=subjectMaskFromBackgroundMask(backgroundMask,w,h);
    const sharpened=sharpenCanvas(source,w,h,amount);
    const sharpSubject=offscreen(w,h);
    const ssctx=sharpSubject.getContext('2d');
    ssctx.drawImage(sharpened,0,0);
    ssctx.globalCompositeOperation='destination-in';
    ssctx.drawImage(subjectMask,0,0,w,h);
    ssctx.globalCompositeOperation='source-over';

    const out=offscreen(w,h);
    const octx=out.getContext('2d');
    octx.drawImage(source,0,0);
    octx.globalAlpha=Math.min(0.9,amount/24);
    octx.drawImage(sharpSubject,0,0);
    octx.globalAlpha=1;
    return out;
  }

  function values(){
    const strength=Math.max(0,Math.min(1,+$('filterStrength').value/100));
    const preset=PRESETS[currentPreset]||PRESETS.glow;
    const raw={
      bright:+$('bright').value,
      sat:+$('sat').value,
      contrast:+$('contrast').value,
      warmth:+$('warmth').value,
      glow:+$('glow').value,
      subjectPop:+$('subjectPop').value
    };

    // プリセット由来の差だけを「フィルタの強さ」で弱める。
    // 手動で動かした差分は、フィルタ強度に関係なく必ず反映する。
    const effective={
      bright:100+(preset.bright-100)*strength+(raw.bright-preset.bright),
      sat:100+(preset.sat-100)*strength+(raw.sat-preset.sat),
      contrast:100+(preset.contrast-100)*strength+(raw.contrast-preset.contrast),
      warmth:preset.warmth*strength+(raw.warmth-preset.warmth),
      glow:Math.max(0,preset.glow*strength+(raw.glow-preset.glow)),
      subjectPop:Math.max(0,preset.subjectPop*strength+(raw.subjectPop-preset.subjectPop))
    };

    return{
      filterStrength:strength,
      bright:effective.bright/100,
      sat:effective.sat/100,
      contrast:effective.contrast/100,
      warmth:effective.warmth,
      glow:effective.glow,
      aperture:+$('apertureF').value,
      blur:+$('blurStrength').value,
      focus:+$('focusDepth').value/100,
      depth:+$('depthOfField').value/100,
      bokehShape:$('bokehShape').value,
      bokehSize:+$('bokehSize').value,
      bokehIntensity:+$('bokehIntensity').value/100,
      bokehDensity:+$('bokehDensity').value/100,
      bokehFringe:+$('bokehFringe').value/100,
      subjectPop:effective.subjectPop
    };
  }

  function applyBasicAdjustments(src,w,h,v){
    const out=offscreen(w,h), octx=out.getContext('2d');
    octx.drawImage(src,0,0,w,h);

    const bright=v.bright;
    const sat=v.sat;
    const contrast=v.contrast;
    const warmth=v.warmth;

    if(Math.abs(bright-1)<0.0001 && Math.abs(sat-1)<0.0001 && Math.abs(contrast-1)<0.0001 && Math.abs(warmth)<0.0001){
      return out;
    }

    const img=octx.getImageData(0,0,w,h), d=img.data;
    const wr=warmth>0 ? warmth*1.48 : warmth*0.32;
    const wg=warmth>0 ? warmth*0.22 : -warmth*0.07;
    const wb=warmth>0 ? -warmth*1.28 : -warmth*1.48;

    for(let i=0;i<d.length;i+=4){
      let r=d[i]*bright;
      let g=d[i+1]*bright;
      let b=d[i+2]*bright;

      const lum=0.2126*r+0.7152*g+0.0722*b;
      r=lum+(r-lum)*sat;
      g=lum+(g-lum)*sat;
      b=lum+(b-lum)*sat;

      r=(r-127.5)*contrast+127.5;
      g=(g-127.5)*contrast+127.5;
      b=(b-127.5)*contrast+127.5;

      r+=wr; g+=wg; b+=wb;
      d[i]=clamp255(r);
      d[i+1]=clamp255(g);
      d[i+2]=clamp255(b);
    }
    octx.putImageData(img,0,0);
    return out;
  }

  function applyPresetLook(out,w,h,strength){
    if(strength<=0) return out;
    const octx=out.getContext('2d');
    octx.save();
    octx.globalAlpha=strength;
    if(currentPreset==='bloom'){
      octx.globalCompositeOperation='soft-light';
      const g=octx.createLinearGradient(0,0,w,h);
      g.addColorStop(0,'rgba(255,214,214,.13)');
      g.addColorStop(1,'rgba(255,240,200,.12)');
      octx.fillStyle=g;octx.fillRect(0,0,w,h);
    }else if(currentPreset==='winter'){
      octx.globalCompositeOperation='soft-light';
      const g=octx.createLinearGradient(0,0,w,h);
      g.addColorStop(0,'rgba(168,208,248,.18)');
      g.addColorStop(.52,'rgba(188,215,248,.15)');
      g.addColorStop(1,'rgba(214,225,245,.10)');
      octx.fillStyle=g;octx.fillRect(0,0,w,h);
      octx.globalCompositeOperation='screen';
      octx.fillStyle='rgba(196,222,255,.045)';octx.fillRect(0,0,w,h);
    }else if(currentPreset==='wooton'){
      octx.globalCompositeOperation='soft-light';
      const g=octx.createLinearGradient(0,0,w,h);
      g.addColorStop(0,'rgba(170,92,44,.18)');
      g.addColorStop(.48,'rgba(218,132,62,.14)');
      g.addColorStop(1,'rgba(246,191,102,.09)');
      octx.fillStyle=g;octx.fillRect(0,0,w,h);
      octx.globalCompositeOperation='multiply';
      octx.fillStyle='rgba(126,80,48,.035)';octx.fillRect(0,0,w,h);
    }else if(currentPreset==='sakura'){
      octx.globalCompositeOperation='soft-light';
      const g=octx.createLinearGradient(0,0,w,h);
      g.addColorStop(0,'rgba(255,168,196,.24)');
      g.addColorStop(.52,'rgba(255,191,214,.19)');
      g.addColorStop(1,'rgba(248,216,230,.12)');
      octx.fillStyle=g;octx.fillRect(0,0,w,h);
      octx.globalCompositeOperation='screen';
      octx.fillStyle='rgba(255,228,242,.085)';octx.fillRect(0,0,w,h);
    }else if(currentPreset==='glow'){
      octx.globalCompositeOperation='screen';
      octx.fillStyle='rgba(255,255,255,.10)';octx.fillRect(0,0,w,h);
    }
    octx.restore();
    return out;
  }

  function applyFilter(src,w,h){
    const v=values();
    const out=applyBasicAdjustments(src,w,h,v);
    applyPresetLook(out,w,h,v.filterStrength);

    if(v.glow>0){
      const octx=out.getContext('2d');
      const glow=offscreen(w,h), gctx=glow.getContext('2d');
      gctx.filter=`blur(${Math.max(2,v.glow*.7)}px)`;
      gctx.drawImage(out,0,0);
      octx.globalCompositeOperation='screen';
      octx.globalAlpha=Math.min(.48,v.glow/90);
      octx.drawImage(glow,0,0);
      octx.globalAlpha=1;
      octx.globalCompositeOperation='source-over';
    }
    return out;
  }

  function composeProcessed(src,w,h,mask,interactive=false){
    const filtered=applyFilter(src,w,h);
    const v=values();
    let out=filtered;

    if($('blurOn').checked && mask){
      // AI深度マップがある場合は WebGL2 の深度依存 Circle-of-Confusion 方式を優先。
      // 人物マスク外だけをサンプリングし、輪郭への背景色の滲みを抑える。
      if(aiDepthCanvas){
        const depthOut=applyDepthAwareDofWebGL(filtered,aiDepthCanvas,mask,w,h,v);
        if(depthOut) out=depthOut;
      }

      // WebGL2非対応 / 深度推定前は、従来の軽量プログレッシブ方式へフォールバック。
      if(out===filtered){
        const merged=offscreen(w,h);
        const mctx=merged.getContext('2d');
        mctx.drawImage(filtered,0,0);
        const aperture=apertureStrength(v.aperture);
        const maxRadius=Math.max(0.6,v.blur*(0.18+aperture*1.22));
        if(v.depth<=0.001){
          const blurred=buildScaleBlur(filtered,w,h,maxRadius);
          const blurLayer=offscreen(w,h), bctx=blurLayer.getContext('2d');
          bctx.drawImage(blurred,0,0);
          bctx.globalCompositeOperation='destination-in';
          bctx.drawImage(mask,0,0,w,h);
          bctx.globalCompositeOperation='source-over';
          mctx.drawImage(blurLayer,0,0);
        }else{
          const depthInfo=buildProgressiveDepthLayers(mask,w,h,v.depth);
          if(depthInfo){
            const radii=[0.22,0.42,0.68,1.00].map(f=>Math.max(0.45,maxRadius*f));
            for(let k=0;k<4;k++){
              const blurred=buildScaleBlur(filtered,w,h,radii[k]);
              const layer=offscreen(w,h), lctx=layer.getContext('2d');
              lctx.drawImage(blurred,0,0);
              lctx.globalCompositeOperation='destination-in';
              lctx.imageSmoothingEnabled=true;
              if('imageSmoothingQuality' in lctx) lctx.imageSmoothingQuality='high';
              lctx.drawImage(depthInfo.layers[k],0,0,depthInfo.sw,depthInfo.sh,0,0,w,h);
              lctx.globalCompositeOperation='source-over';
              mctx.drawImage(layer,0,0);
            }
          }
        }
        out=merged;
      }

      if(aiDepthCanvas && out){
        out=applyBokehShapes(out,filtered,aiDepthCanvas,mask,w,h,v,interactive);
      }
    }

    // これは既存の「被写体くっきり」調整。背景ぼかし自体は人物の画素を変更しない。
    if(!interactive && v.subjectPop>0 && mask){
      out=applySubjectPop(out,w,h,mask,v.subjectPop);
    }
    return out;
  }

  function render(mode='full'){
    if(!srcCanvas) return;
    const usePreview=mode==='preview';
    ctx.clearRect(0,0,workW,workH);
    ctx.imageSmoothingEnabled=true;
    if('imageSmoothingQuality' in ctx) ctx.imageSmoothingQuality=usePreview?'medium':'high';
    if(!showAfter){
      ctx.drawImage(srcCanvas,0,0);
      return;
    }
    const sourceInfo=usePreview ? getPreviewSource() : {canvas:srcCanvas,w:workW,h:workH};
    const effective=aiDepthCanvas?buildLensBackgroundMask(sourceInfo.w,sourceInfo.h):buildEffectiveBackgroundMask(sourceInfo.w,sourceInfo.h);
    const out=composeProcessed(sourceInfo.canvas,sourceInfo.w,sourceInfo.h,effective,usePreview);
    if(usePreview && (sourceInfo.w!==workW || sourceInfo.h!==workH)) ctx.drawImage(out,0,0,sourceInfo.w,sourceInfo.h,0,0,workW,workH);
    else ctx.drawImage(out,0,0);
  }

  function requestRender(mode='full'){
    if(mode==='full') pendingRenderMode='full';
    else if(!renderScheduled) pendingRenderMode='preview';
    else if(pendingRenderMode!=='full') pendingRenderMode='preview';
    if(renderScheduled) return;
    renderScheduled=true;
    requestAnimationFrame(()=>{
      renderScheduled=false;
      const nextMode=pendingRenderMode;
      pendingRenderMode='full';
      render(nextMode);
    });
  }

  function requestInteractiveRender(){
    requestRender('preview');
    clearTimeout(settleRenderTimer);
    settleRenderTimer=setTimeout(()=>requestRender('full'),260);
  }

  function updateViewButtons(){
    $('btnBeforeStage').classList.toggle('active',!showAfter);
    $('btnAfterStage').classList.toggle('active',showAfter);
  }

  function updateCorrectionUI(){
    const on=$('correctionOn').checked;
    canvasWrap.classList.toggle('is-correcting',on);
    correctionBanner.style.display=on?'flex':'none';
    correctionModeLabel.textContent=correctionMode==='restore'?'現在: 鮮明に戻す':'現在: ぼかしを足す';
    $('restoreBtn').classList.toggle('active',correctionMode==='restore');
    $('addBlurBtn').classList.toggle('active',correctionMode==='add');
  }


  function applyPreset(name){currentPreset=name;const p=PRESETS[name];Object.keys(p).forEach(k=>{$(k).value=p[k];$(k+'Out').textContent=p[k];});document.querySelectorAll('#presetWrap button').forEach(b=>b.classList.toggle('active',b.dataset.preset===name));requestRender('full');}
  document.querySelectorAll('#presetWrap button').forEach(btn=>btn.addEventListener('click',()=>applyPreset(btn.dataset.preset)));
  $('btnResetPreset').addEventListener('click',()=>applyPreset(currentPreset));
  document.querySelectorAll('[data-group-toggle]').forEach(btn=>btn.addEventListener('click',()=>{
    const group=btn.closest('.group');
    const willOpen=!group.classList.contains('open');
    group.classList.toggle('open',willOpen);
    btn.setAttribute('aria-expanded',willOpen?'true':'false');
  }));
  applyPreset('glow');
  updateViewButtons();
  updateCorrectionUI();

  fileInput.addEventListener('change',async()=>{
    const file=fileInput.files&&fileInput.files[0];if(!file)return;
    setStatus(`読み込み中…\n${file.name}\n${file.type||'MIME不明'} / ${Math.round(file.size/1024)} KB`);
    try{
      const decoded=await decodeFile(file);fullImg=decoded;fullW=decoded.width||decoded.naturalWidth;fullH=decoded.height||decoded.naturalHeight;if(!fullW||!fullH)throw new Error('画像サイズを取得できませんでした');
      const scale=Math.min(1,WORK_MAX/Math.max(fullW,fullH));workW=Math.max(1,Math.round(fullW*scale));workH=Math.max(1,Math.round(fullH*scale));
      srcCanvas=offscreen(workW,workH);
      srcCanvas.getContext('2d').drawImage(fullImg,0,0,workW,workH);
      canvas.width=workW;canvas.height=workH;
      maskPaintCanvas.width=workW;maskPaintCanvas.height=workH;
      maskPaintCtx.clearRect(0,0,workW,workH);
      invalidatePreviewCache();
      effectiveMaskCache.clear();
      lensMaskCache.clear();
      aiDepthCanvas=null;depthError=null;depthAutoFocused=false;depthManualInvert=false;depthOrientationChecked=false;aiDepthRevision++;updateDepthUI();
        maskRevision=0;
      addBlurMask=null;
      restoreSharpMask=null;
      painting=false;
      lastPaint=null;
      pickerCard.style.display='none';canvasWrap.style.display='block';panel.style.display='block';viewerTools.style.display='block';stickySavebar.style.display='block';
      autoBackgroundMask=null;segError=null;updateSegUI();
      applyPreset(currentPreset);
      computeAutoBackgroundMask();
      computeAIDepthMap();
    }catch(err){setStatus('読み込み失敗\n'+(err&&err.message?err.message:String(err))+'\n\nHEICの場合はJPEG/PNGで試してください。');}
  });

  $('btnBeforeStage').addEventListener('click',()=>{showAfter=false;updateViewButtons();requestRender('full');});
  $('btnAfterStage').addEventListener('click',()=>{showAfter=true;updateViewButtons();requestRender('full');});
  $('blurOn').addEventListener('change',()=>requestRender('full'));
  $('bokehShape').addEventListener('change',()=>{showAfter=true;updateViewButtons();requestRender('full');});
  const visualControlIds=new Set(['filterStrength','bright','sat','contrast','warmth','glow','subjectPop','bokehSize','bokehIntensity','bokehDensity','bokehFringe']);
  function ensureAfterViewForAdjustment(id){
    if(!visualControlIds.has(id)) return;
    if(!showAfter){
      showAfter=true;
      updateViewButtons();
    }
  }
  sliderIds.forEach(id=>{
    const el=$(id);
    const preview=()=>{ensureAfterViewForAdjustment(id);requestInteractiveRender();};
    const finish=()=>{ensureAfterViewForAdjustment(id);requestRender('full');};
    el.addEventListener('input',preview);
    el.addEventListener('change',finish);
    el.addEventListener('pointerup',finish);
    el.addEventListener('touchend',finish,{passive:true});
  });
  $('focusSubjectBtn').addEventListener('click',()=>{if(sampleDepthForSubject(true)===null) showToast('人物とAI深度マップの準備後に使えます。','info');});
  $('toggleDepthPreview').addEventListener('click',()=>{
    if(!aiDepthCanvas){showToast('AI深度マップを生成中です。','info');return;}
    const show=depthPreviewWrap.style.display==='none';
    depthPreviewWrap.style.display=show?'block':'none';
    $('toggleDepthPreview').textContent=show?'深度マップを隠す':'深度マップを見る';
    if(show) drawDepthPreview();
  });
  depthPreviewCanvas.addEventListener('click',()=>{
    if(!aiDepthCanvas) return;
    depthManualInvert=!depthManualInvert;depthOrientationChecked=true;
    invertDepthCanvas(aiDepthCanvas);aiDepthRevision++;drawDepthPreview();requestRender('full');
    showToast('深度マップを反転しました。','info');
  });
  $('correctionOn').addEventListener('change',()=>{if(!$('correctionOn').checked){painting=false;lastPaint=null;paintRect=null;queuedPaintPoint=null;if(paintRafId){cancelAnimationFrame(paintRafId);paintRafId=0;}maskPaintCtx.clearRect(0,0,workW,workH);}updateCorrectionUI();});
  $('restoreBtn').addEventListener('click',()=>{correctionMode='restore';updateCorrectionUI();});
  $('addBlurBtn').addEventListener('click',()=>{correctionMode='add';updateCorrectionUI();});
  $('clearCorrection').addEventListener('click',()=>{queuedPaintPoint=null;if(paintRafId){cancelAnimationFrame(paintRafId);paintRafId=0;}if(addBlurMask)addBlurMask.getContext('2d').clearRect(0,0,workW,workH);if(restoreSharpMask)restoreSharpMask.getContext('2d').clearRect(0,0,workW,workH);maskPaintCtx.clearRect(0,0,workW,workH);bumpMaskRevision();requestRender('full');});

  // 補正ブラシ軽量化：ドラッグ中は重い画像再処理を行わず、
  // マスクと軽いブラシ軌跡だけ更新。加工結果は指を離した時に1回だけ再描画する。
  function beginPaint(clientX,clientY){
    paintRect=maskPaintCanvas.getBoundingClientRect();
    paintScaleX=workW/paintRect.width;
    paintScaleY=workH/paintRect.height;
    const avgScale=(paintScaleX+paintScaleY)/2;
    paintRadius=(+$('brushSize').value/2)*avgScale;
    const trackingMode=$('brushTrackingMode').value;
    paintMinDistance=trackingMode==='precise'
      ? Math.max(0.8,Math.min(2.2,paintRadius*0.06))
      : Math.max(1.5,Math.min(6,paintRadius*0.16));
    queuedPaintPoint=null;
    if(paintRafId){cancelAnimationFrame(paintRafId);paintRafId=0;}
    lastPaint=null;
    maskPaintCtx.clearRect(0,0,workW,workH);
    paintPoint(clientX,clientY);
  }

  function drawBrushPreview(x,y,prev){
    maskPaintCtx.save();
    maskPaintCtx.strokeStyle=correctionMode==='restore'?'rgba(120,220,160,.42)':'rgba(230,190,105,.42)';
    maskPaintCtx.fillStyle=maskPaintCtx.strokeStyle;
    maskPaintCtx.lineWidth=paintRadius*2;
    maskPaintCtx.lineCap='round';
    maskPaintCtx.lineJoin='round';
    if(prev){
      maskPaintCtx.beginPath();
      maskPaintCtx.moveTo(prev.x,prev.y);
      maskPaintCtx.lineTo(x,y);
      maskPaintCtx.stroke();
    }else{
      maskPaintCtx.beginPath();
      maskPaintCtx.arc(x,y,paintRadius,0,Math.PI*2);
      maskPaintCtx.fill();
    }
    maskPaintCtx.restore();
  }

  function paintPoint(clientX,clientY){
    if(!autoBackgroundMask || !paintRect) return;
    const x=(clientX-paintRect.left)*paintScaleX;
    const y=(clientY-paintRect.top)*paintScaleY;
    const prev=lastPaint;
    if(prev){
      const dx=x-prev.x, dy=y-prev.y;
      if((dx*dx+dy*dy) < (paintMinDistance*paintMinDistance)) return;
    }
    const target=correctionMode==='add'?addBlurMask:restoreSharpMask;
    const tctx=target.getContext('2d');
    tctx.strokeStyle='rgba(255,255,255,1)';
    tctx.fillStyle='rgba(255,255,255,1)';
    tctx.lineWidth=paintRadius*2;
    tctx.lineCap='round';
    tctx.lineJoin='round';
    if(prev){
      tctx.beginPath();
      tctx.moveTo(prev.x,prev.y);
      tctx.lineTo(x,y);
      tctx.stroke();
    }else{
      tctx.beginPath();
      tctx.arc(x,y,paintRadius,0,Math.PI*2);
      tctx.fill();
    }
    drawBrushPreview(x,y,prev);
    lastPaint={x,y};
  }

  function queuePaint(clientX,clientY){
    queuedPaintPoint={clientX,clientY};
    if(paintRafId) return;
    paintRafId=requestAnimationFrame(()=>{
      paintRafId=0;
      if(!queuedPaintPoint) return;
      const pt=queuedPaintPoint;
      queuedPaintPoint=null;
      paintPoint(pt.clientX,pt.clientY);
    });
  }

  function finishPaint(){
    if(!painting) return;
    painting=false;
    if(paintRafId){cancelAnimationFrame(paintRafId);paintRafId=0;}
    if(queuedPaintPoint){const pt=queuedPaintPoint;queuedPaintPoint=null;paintPoint(pt.clientX,pt.clientY);}
    lastPaint=null;
    paintRect=null;
    maskPaintCtx.clearRect(0,0,workW,workH);
    bumpMaskRevision();
    requestInteractiveRender();
  }

  maskPaintCanvas.addEventListener('pointerdown',e=>{
    if(!$('correctionOn').checked)return;
    e.preventDefault();
    painting=true;
    maskPaintCanvas.setPointerCapture(e.pointerId);
    beginPaint(e.clientX,e.clientY);
  });
  maskPaintCanvas.addEventListener('pointermove',e=>{
    if(!painting)return;
    e.preventDefault();
    if($('brushTrackingMode').value==='precise') paintPoint(e.clientX,e.clientY);
    else queuePaint(e.clientX,e.clientY);
  });
  maskPaintCanvas.addEventListener('pointerup',finishPaint);
  maskPaintCanvas.addEventListener('pointercancel',finishPaint);

  $('btnNew').addEventListener('click',()=>{fileInput.value='';fileInput.click();});

  function roundRect(c,x,y,w,h,r){c.beginPath();c.moveTo(x+r,y);c.arcTo(x+w,y,x+w,y+h,r);c.arcTo(x+w,y+h,x,y+h,r);c.arcTo(x,y+h,x,y,r);c.arcTo(x,y,x+w,y,r);c.closePath();}

  function drawCopyrightWatermark(octx,w,h){
    if(!$('copyrightOn').checked) return;
    const scale=(+$('copyrightScale').value||100)/100;
    const opacity=(+$('copyrightOpacity').value||100)/100;
    const pad=Math.max(18,Math.round(w*.014*scale));
    const fontSize=Math.max(16,Math.round(w*.018*scale));
    const text='(C) SQUARE ENIX';
    const styleEl=$('copyrightStyle');
    const style=styleEl ? styleEl.value : 'soft-panel';
    const pos=$('copyrightPos').value;
    const offsetX=+$('copyrightOffsetX').value||0;
    const offsetY=+$('copyrightOffsetY').value||0;

    octx.save();
    octx.globalAlpha=Math.max(.2,Math.min(1,opacity));
    octx.font=`600 ${fontSize}px -apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic",sans-serif`;
    octx.textAlign='left';
    octx.textBaseline='bottom';
    octx.lineJoin='round';

    const tw=octx.measureText(text).width;
    const panelStyle=style==='soft-panel'||style==='light-panel';
    const hPad=panelStyle?fontSize*.46:fontSize*.22;
    const vPad=panelStyle?fontSize*.36:fontSize*.24;
    const bw=tw+hPad*2;
    const bh=fontSize+vPad*2;

    let boxX=pad;
    let boxY=h-pad-bh;
    if(pos==='bottom-right') boxX=w-pad-bw;
    else if(pos==='bottom-center') boxX=Math.round((w-bw)/2);
    else if(pos==='top-right') {boxX=w-pad-bw;boxY=pad;}
    else if(pos==='top-left') {boxX=pad;boxY=pad;}
    else if(pos==='top-center') {boxX=Math.round((w-bw)/2);boxY=pad;}

    boxX=Math.max(pad,Math.min(w-pad-bw,boxX+offsetX));
    boxY=Math.max(pad,Math.min(h-pad-bh,boxY+offsetY));
    const textX=boxX+hPad;
    const textY=boxY+bh-vPad;

    if(style==='soft-panel'){
      octx.fillStyle='rgba(0,0,0,.34)';
      roundRect(octx,boxX,boxY,bw,bh,fontSize*.52);
      octx.fill();
      octx.fillStyle='rgba(255,255,255,.97)';
      octx.fillText(text,textX,textY);
    }else if(style==='light-panel'){
      octx.fillStyle='rgba(255,255,255,.76)';
      roundRect(octx,boxX,boxY,bw,bh,fontSize*.52);
      octx.fill();
      octx.fillStyle='rgba(27,48,68,.95)';
      octx.fillText(text,textX,textY);
    }else if(style==='white-shadow'){
      octx.shadowColor='rgba(0,0,0,.72)';
      octx.shadowBlur=Math.max(2,fontSize*.18);
      octx.shadowOffsetY=Math.max(1,fontSize*.07);
      octx.fillStyle='rgba(255,255,255,.98)';
      octx.fillText(text,textX,textY);
    }else if(style==='white-outline'){
      octx.lineWidth=Math.max(2,fontSize*.11);
      octx.strokeStyle='rgba(0,0,0,.78)';
      octx.strokeText(text,textX,textY);
      octx.fillStyle='rgba(255,255,255,.99)';
      octx.fillText(text,textX,textY);
    }else if(style==='black-outline'){
      octx.lineWidth=Math.max(2,fontSize*.12);
      octx.strokeStyle='rgba(255,255,255,.92)';
      octx.strokeText(text,textX,textY);
      octx.fillStyle='rgba(16,22,29,.96)';
      octx.fillText(text,textX,textY);
    }else{
      octx.fillStyle='rgba(255,255,255,.97)';
      octx.fillText(text,textX,textY);
    }
    octx.restore();
  }

  function sanitizeFileName(name){
    const cleaned=(name||'palette_studio')
      .replace(/[\\/:*?\"<>|]+/g,'_')
      .replace(/^\.+|\.+$/g,'')
      .trim();
    return cleaned || 'palette_studio';
  }

  function getExportSpec(){
    const requested=$('saveSize').value;
    let w=fullW, h=fullH;
    if(requested!=='original'){
      const maxSide=+requested;
      const scale=Math.min(1,maxSide/Math.max(fullW,fullH));
      w=Math.max(1,Math.round(fullW*scale));
      h=Math.max(1,Math.round(fullH*scale));
    }
    const format=$('saveFormat').value;
    const mime=format==='jpeg'?'image/jpeg':'image/png';
    const ext=format==='jpeg'?'jpg':'png';
    const quality=format==='jpeg'?(+$('jpegQuality').value/100):undefined;
    const fileName=`${sanitizeFileName($('saveFileName').value)}.${ext}`;
    return {w,h,format,mime,ext,quality,fileName};
  }

  async function exportImage(){
    const spec=getExportSpec();
    const exportSrc=offscreen(spec.w,spec.h);
    const esctx=exportSrc.getContext('2d');
    esctx.imageSmoothingEnabled=true;
    if('imageSmoothingQuality' in esctx) esctx.imageSmoothingQuality='high';
    esctx.drawImage(fullImg,0,0,fullW,fullH,0,0,spec.w,spec.h);

    let exportMask=null;
    if(autoBackgroundMask) exportMask=buildEffectiveBackgroundMask(spec.w,spec.h);
    const out=composeProcessed(exportSrc,spec.w,spec.h,exportMask);
    const octx=out.getContext('2d');
    if($('copyrightOn').checked) drawCopyrightWatermark(octx,spec.w,spec.h);

    const blob=await new Promise(resolve=>out.toBlob(resolve,spec.mime,spec.quality));
    if(!blob) throw new Error('画像の書き出しに失敗しました');
    return {blob,spec};
  }

  function setSaveBusy(busy,label='保存データを作成中…'){
    $('btnSave').disabled=busy;
    $('btnShare').disabled=busy;
    $('btnNew').disabled=busy;
    btnSaveSticky.disabled=busy;
    btnShareSticky.disabled=busy;
    manualOverlayBusy=busy;
    manualOverlayTitle=label;
    manualOverlaySub=busy?'高画質で画像を書き出しています。処理が終わるまでそのままお待ちください。':'';
    if(busy){
      $('btnSave').textContent=label;
      $('btnShare').textContent=label;
      btnSaveSticky.textContent=label;
      btnShareSticky.textContent=label;
    }else{
      updateSaveUI();
      $('btnShare').textContent='共有メニューを開く';
      btnShareSticky.textContent='共有メニュー';
    }
    updateOverlayFromState();
  }

  function setSaveStatus(msg){$('saveStatus').textContent=msg;}

  function showToast(msg,type='info'){
    if(!toastEl || !toastText) return;
    toastText.textContent=msg;
    toastEl.className=`toast ${type}`;
    toastEl.classList.add('show');
    if(toastTimer) clearTimeout(toastTimer);
    toastTimer=setTimeout(()=>toastEl.classList.remove('show'),2400);
  }

  function updateSaveUI(){
    const isJpeg=$('saveFormat').value==='jpeg';
    const saveLabel=isJpeg?'JPEGを端末に保存':'PNGを端末に保存';
    $('jpegQualityRow').style.display=isJpeg?'flex':'none';
    $('btnSave').textContent=saveLabel;
    btnSaveSticky.textContent=isJpeg?'JPEG保存':'PNG保存';
    $('btnShare').textContent='共有メニューを開く';
    btnShareSticky.textContent='共有メニュー';
    $('saveStatus').textContent=isJpeg
      ? 'JPEGは軽く共有しやすい形式です。iPhoneで写真アプリへ入れたい時は「共有メニューを開く」→「画像を保存」を選んでください。'
      : 'PNGはきれいに保存したい時におすすめです。iPhoneで写真アプリへ入れたい時は「共有メニューを開く」→「画像を保存」を選んでください。';
  }

  async function ensureMaskReadyForSave(){
    if(!$('blurOn').checked) return;
    if(!autoBackgroundMask){
      if(segLoading){
        while(segLoading) await new Promise(r=>setTimeout(r,60));
      }else{
        await computeAutoBackgroundMask();
      }
    }
    if(!aiDepthCanvas && !depthError){
      if(depthComputePromise) await depthComputePromise;
      else await computeAIDepthMap();
    }
  }

  async function saveByDownload(){
    if(!fullImg) return;
    setSaveBusy(true);
    setSaveStatus('保存用データを作成しています…');
    try{
      await ensureMaskReadyForSave();
      await new Promise(requestAnimationFrame);
      const {blob,spec}=await exportImage();
      const url=URL.createObjectURL(blob);
      const a=document.createElement('a');
      a.href=url;
      a.download=spec.fileName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(()=>URL.revokeObjectURL(url),2500);
      setSaveStatus(`${spec.fileName} を端末に保存しました。`);
      showToast('端末に保存しました。','success');
    }catch(err){
      console.error(err);
      setSaveStatus('端末への保存に失敗しました。もう一度お試しください。');
      showToast('保存に失敗しました。','error');
    }finally{
      setSaveBusy(false);
    }
  }

  async function saveByShare(){
    if(!fullImg) return;
    setSaveBusy(true,'共有用データを作成中…');
    setSaveStatus('共有メニュー用の画像を準備しています…');
    try{
      await ensureMaskReadyForSave();
      await new Promise(requestAnimationFrame);
      const {blob,spec}=await exportImage();
      const file=new File([blob],spec.fileName,{type:spec.mime});
      if(navigator.share && (!navigator.canShare || navigator.canShare({files:[file]}))){
        await navigator.share({files:[file],title:'Palette Studio'});
        setSaveStatus('共有メニューを開きました。iPhoneでは「画像を保存」を選ぶと写真アプリに入れられます。');
        showToast('共有メニューを開きました。','success');
      }else{
        const url=URL.createObjectURL(blob);
        const a=document.createElement('a');
        a.href=url;
        a.download=spec.fileName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(()=>URL.revokeObjectURL(url),2500);
        setSaveStatus('このブラウザでは共有に対応していないため、端末保存に切り替えました。');
        showToast('共有未対応のため端末保存に切り替えました。','info');
      }
    }catch(err){
      if(err && err.name==='AbortError') {setSaveStatus('共有をキャンセルしました。'); showToast('共有をキャンセルしました。','info');}
      else {console.error(err);setSaveStatus('共有メニュー用画像の作成に失敗しました。'); showToast('共有の準備に失敗しました。','error');}
    }finally{
      setSaveBusy(false);
    }
  }

  $('saveFormat').addEventListener('change',updateSaveUI);
  $('jpegQuality').addEventListener('input',()=>{$('jpegQualityOut').textContent=$('jpegQuality').value;});
  updateSaveUI();



  window.addEventListener('error',e=>{
    console.error(e.error||e.message);
    if(!fullImg) setStatus('JavaScriptエラー: '+(e.message||'不明なエラー'));
  });

  $('btnSave').addEventListener('click',saveByDownload);
  $('btnShare').addEventListener('click',saveByShare);
  btnSaveSticky.addEventListener('click',saveByDownload);
  btnShareSticky.addEventListener('click',saveByShare);
})();
