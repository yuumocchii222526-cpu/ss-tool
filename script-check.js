
(() => {
  const $ = id => document.getElementById(id);
  const MEDIAPIPE_VERSION = '0.10.14';
  const fileInput=$('fileInput'), pickerCard=$('pickerCard'), status=$('status'), panel=$('panel'), canvasWrap=$('canvasWrap'), canvas=$('canvas'), ctx=canvas.getContext('2d');
  const segDot=$('segDot'), segStatus=$('segStatus');
  const viewerTools=$('viewerTools'), correctionBanner=$('correctionBanner'), correctionModeLabel=$('correctionModeLabel');
  const processingOverlay=$('processingOverlay'), overlayText=$('overlayText'), overlaySub=$('overlaySub');
  const stickySavebar=$('stickySavebar'), btnSaveSticky=$('btnSaveSticky'), btnShareSticky=$('btnShareSticky');
  const toastEl=$('toast'), toastText=$('toastText');
  const sliderIds=['blurStrength','depthOfField','brushSize','filterStrength','bright','sat','contrast','warmth','glow','subjectPop','copyrightOffsetX','copyrightOffsetY','copyrightScale','copyrightOpacity'];
  sliderIds.forEach(id=>{const el=$(id), out=$(id+'Out'); out.textContent=el.value; el.addEventListener('input',()=>out.textContent=el.value);});

  let fullImg=null, fullW=0, fullH=0, workW=0, workH=0, srcCanvas=null;
  let showAfter=true, currentPreset='glow';
  let imageSegmenterInstance=null, autoBackgroundMask=null, segLoading=false, segError=null;

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
  let depthMaskCacheKey='';
  let depthMaskCacheValue=null;
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

  function invalidatePreviewCache(){
    previewSourceCache=null;
  }

  function bumpMaskRevision(){
    maskRevision++;
    effectiveMaskCache.clear();
    depthMaskCacheKey='';
    depthMaskCacheValue=null;
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
    }finally{segLoading=false;updateSegUI();render();}
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

    // 1) 背景ぼかし量は保ちつつ、人物境界だけ不自然になりにくいよう少しだけ圧縮
    let refined=remapBackgroundMaskAlpha(baseMask,w,h,0.34,0.80);

    // 2) 人物の縁に「やわらかい保護帯」を作って、手や腕を守りながら境界はなだらかにする
    const subjectMask=subjectMaskFromBackgroundMask(refined,w,h);
    const protectGrow=Math.max(1,Math.round(Math.max(w,h)/420));
    const expandedSubject=expandAlphaMask(subjectMask,w,h,protectGrow);
    const protectBlur=Math.max(4,Math.round(Math.max(w,h)/170));
    const protect=offscreen(w,h), pctx=protect.getContext('2d');
    pctx.filter=`blur(${protectBlur}px)`;
    pctx.drawImage(expandedSubject,0,0,w,h);
    pctx.filter='none';
    const protectImg=pctx.getImageData(0,0,w,h), pd=protectImg.data;
    for(let i=0;i<pd.length;i+=4){
      const a=pd[i+3]/255;
      const na=smoothstep(0.10,0.52,a);
      pd[i]=255; pd[i+1]=255; pd[i+2]=255; pd[i+3]=Math.round(na*255);
    }
    pctx.putImageData(protectImg,0,0);

    const cut=offscreen(w,h), cctx=cut.getContext('2d');
    cctx.drawImage(refined,0,0,w,h);
    cctx.globalCompositeOperation='destination-out';
    cctx.drawImage(protect,0,0,w,h);
    cctx.globalCompositeOperation='source-over';

    // 3) 境目を少し広めにフェザーして、背景のぼけ感は残したまま切り抜き感を減らす
    refined=softenMaskEdge(cut,w,h,2.4);
    refined=remapBackgroundMaskAlpha(refined,w,h,0.06,0.98);
    return refined;
  }

  function buildEffectiveBackgroundMask(w,h){
    if(!autoBackgroundMask && !addBlurMask) return null;
    const cacheKey=`${maskRevision}:${w}x${h}`;
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


  // 疑似的な被写界深度マスク。
  // AIの距離推定ではなく、人物マスクからの画面上の距離を使い、
  // 人物に近い背景ほどピントを残し、離れた背景ほど強くぼかす。
  function buildDepthFarMask(backgroundMask,w,h,depth){
    const d=clamp01(depth);
    if(!backgroundMask || d<=0.001) return backgroundMask;

    const depthStep=Math.round(d*100);
    const cacheKey=`${maskRevision}:${w}x${h}:${depthStep}`;
    if(depthMaskCacheKey===cacheKey && depthMaskCacheValue) return depthMaskCacheValue;

    // 大きな画像でもiPhoneで重くなりすぎないよう、保護帯は小さいマスクで作る。
    const maxSide=560;
    const scale=Math.min(1,maxSide/Math.max(w,h));
    const sw=Math.max(2,Math.round(w*scale));
    const sh=Math.max(2,Math.round(h*scale));

    const bgSmall=offscreen(sw,sh), bgctx=bgSmall.getContext('2d');
    bgctx.imageSmoothingEnabled=true;
    if('imageSmoothingQuality' in bgctx) bgctx.imageSmoothingQuality='medium';
    bgctx.drawImage(backgroundMask,0,0,w,h,0,0,sw,sh);

    const subjectSmall=subjectMaskFromBackgroundMask(bgSmall,sw,sh);

    // 人物の周囲を今までより広めに保護し、
    // 被写界深度を上げたときに「人物の近くの背景」までぼけにくくする。
    const expandedRadius=Math.max(2,Math.round((Math.max(w,h)*(0.012 + d*0.030))*scale));
    const expandedSubject=expandAlphaMask(subjectSmall,sw,sh,expandedRadius);
    const protectSmall=offscreen(sw,sh), pctx=protectSmall.getContext('2d');

    // 深くするほどピントが残る帯を広げる。
    const fullRadius=Math.max(w,h)*(0.050 + d*0.230);
    const smallRadius=Math.max(4,Math.min(76,fullRadius*scale));
    pctx.filter=`blur(${smallRadius}px)`;
    pctx.drawImage(expandedSubject,0,0,sw,sh);
    pctx.filter='none';

    // ガウスぼかしの薄い裾まで使って、境界をなだらかにする。
    const img=pctx.getImageData(0,0,sw,sh), px=img.data;
    const thresholdHigh=Math.max(0.045,0.13 - d*0.08);
    for(let i=0;i<px.length;i+=4){
      const a=px[i+3]/255;
      const protect=smoothstep(0.004,thresholdHigh,a);
      px[i]=255;px[i+1]=255;px[i+2]=255;px[i+3]=Math.round(protect*255);
    }
    pctx.putImageData(img,0,0);

    const protectFull=offscreen(w,h), pfctx=protectFull.getContext('2d');
    pfctx.imageSmoothingEnabled=true;
    if('imageSmoothingQuality' in pfctx) pfctx.imageSmoothingQuality='high';
    pfctx.drawImage(protectSmall,0,0,sw,sh,0,0,w,h);

    const far=offscreen(w,h), fctx=far.getContext('2d');
    fctx.drawImage(backgroundMask,0,0,w,h);
    fctx.globalCompositeOperation='destination-out';
    fctx.drawImage(protectFull,0,0,w,h);
    fctx.globalCompositeOperation='source-over';

    // 人物付近の背景が完全にぼけ側へ流れ込みにくいよう、
    // 0に近い薄いアルファは切り落として far 領域を整理する。
    const farImg=fctx.getImageData(0,0,w,h), fd=farImg.data;
    for(let i=0;i<fd.length;i+=4){
      const a=fd[i+3]/255;
      let na=smoothstep(0.018,0.98,a);
      if(na<0.008) na=0;
      fd[i]=255; fd[i+1]=255; fd[i+2]=255; fd[i+3]=Math.round(na*255);
    }
    fctx.putImageData(farImg,0,0);

    depthMaskCacheKey=cacheKey;
    depthMaskCacheValue=far;
    return far;
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
      blur:+$('blurStrength').value,
      depth:+$('depthOfField').value/100,
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
      const merged=offscreen(w,h);
      const mctx=merged.getContext('2d');
      mctx.drawImage(filtered,0,0);

      if(v.depth<=0.001){
        // 0は従来と完全に同じ「背景全体を均一にぼかす」動作。
        const blurred=buildScaleBlur(filtered,w,h,v.blur);
        const blurLayer=offscreen(w,h);
        const bctx=blurLayer.getContext('2d');
        bctx.drawImage(blurred,0,0);
        bctx.globalCompositeOperation='destination-in';
        bctx.drawImage(mask,0,0,w,h);
        bctx.globalCompositeOperation='source-over';
        mctx.drawImage(blurLayer,0,0);
      }else{
        // 人物の近くは「できるだけ元画像のまま」残し、
        // 離れた背景だけを強くぼかす。
        // これで人物のすぐ近くまでぼけが乗りにくくなる。
        const baseFactor=Math.max(0,0.42 - v.depth*0.42);
        const baseRadius=v.blur*baseFactor;
        if(baseRadius>=0.35){
          const baseBlur=buildScaleBlur(filtered,w,h,baseRadius);
          const baseLayer=offscreen(w,h), bctx=baseLayer.getContext('2d');
          bctx.drawImage(baseBlur,0,0);
          bctx.globalCompositeOperation='destination-in';
          bctx.drawImage(mask,0,0,w,h);
          bctx.globalCompositeOperation='source-over';
          mctx.drawImage(baseLayer,0,0);
        }

        const farMask=buildDepthFarMask(mask,w,h,v.depth);
        const farBlur=buildScaleBlur(filtered,w,h,v.blur);
        const farLayer=offscreen(w,h), fctx=farLayer.getContext('2d');
        fctx.drawImage(farBlur,0,0);
        fctx.globalCompositeOperation='destination-in';
        fctx.drawImage(farMask,0,0,w,h);
        fctx.globalCompositeOperation='source-over';
        mctx.drawImage(farLayer,0,0);
      }
      out=merged;
    }

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
    const effective=buildEffectiveBackgroundMask(sourceInfo.w,sourceInfo.h);
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
        maskRevision=0;
      addBlurMask=null;
      restoreSharpMask=null;
      painting=false;
      lastPaint=null;
      pickerCard.style.display='none';canvasWrap.style.display='block';panel.style.display='block';viewerTools.style.display='block';stickySavebar.style.display='block';
      autoBackgroundMask=null;segError=null;updateSegUI();
      applyPreset(currentPreset);
      computeAutoBackgroundMask();
    }catch(err){setStatus('読み込み失敗\n'+(err&&err.message?err.message:String(err))+'\n\nHEICの場合はJPEG/PNGで試してください。');}
  });

  $('btnBeforeStage').addEventListener('click',()=>{showAfter=false;updateViewButtons();requestRender('full');});
  $('btnAfterStage').addEventListener('click',()=>{showAfter=true;updateViewButtons();requestRender('full');});
  $('blurOn').addEventListener('change',()=>requestRender('full'));
  const visualControlIds=new Set(['filterStrength','bright','sat','contrast','warmth','glow','subjectPop']);
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
    if($('blurOn').checked && !autoBackgroundMask && !segLoading){
      await computeAutoBackgroundMask();
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
