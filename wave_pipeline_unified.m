% wave_pipeline_unified.m
%
% A single call runs the whole chain for ONE subject or a WHOLE folder tree.
%
% USAGE
%   wave_pipeline_unified('tree', rawRoot,        outRoot)   % whole dataset
%   wave_pipeline_unified('one',  rawSubjectPath, outRoot)   % single subject
%
%   rawRoot        : <root>\<AgeGroup>\<EyeSide>\<Subject>\*.JPG
%   rawSubjectPath : <root>\<AgeGroup>\<EyeSide>\<Subject>  (age/eye/subject
%                    inferred from the last three path components)
%   Both image sides (R / L) are processed for each subject.
%
% PIPELINE OVERVIEW (one side of one subject)
%   1. Read every JPG once, crop to the corneal ROI, optionally mirror L->R.
%   2. Segment each frame into anterior / posterior corneal surfaces.
%   3. Temporally smooth / repair those surfaces across frames.
%   4. Build one arc-length geometry per frame, shared by both branches.
%   5a. "Binary" branch: warp each frame into an arc-uniform rectangle,
%       difference consecutive frames, threshold, keep the wave blobs,
%       then track that blob forward in time.
%   5b. "Profile" branch: sample 1-D intensity profiles perpendicular to
%       the arc, look for the wave's bright peak, scan the obliquity angle,
%       fit a trajectory with RANSAC, refine it, and prune wrong points.
%   6.  Write the per-frame diagnostics and summary.
% ══════════════════════════════════════════════════════════════════════

function [rows, allResults] = wave_pipeline_unified(mode, inPath, outRoot, saveImages)
% Main entry point.
%
% INPUTS
%   mode    : 'tree' (or 'run') processes every subject under inPath.
%             'one'    processes a single subject folder inPath.
%             'verify' runs the synthetic ground-truth verification suite and
%                      needs only an output root (call as
%                      wave_pipeline_unified('verify', outRoot)).
%   inPath  : For 'tree', the dataset root containing <age>\<eye>\<subject>.
%             For 'one', the subject folder (age/eye/subject inferred).
%             For 'verify', the output root (no input data needed).
%   outRoot : Output root. Per-subject results go into
%             <outRoot>\<age>\<eye>\<subject>_<imgSide>_<meridian>\.
%
% OUTPUTS
%   rows       : Cell array of packed result structs, one per (subject, side).
%   allResults : Struct whose fields are the binary tracker outputs,
%                keyed by a sanitized subject identifier. Saved to
%                <outRoot>/all_tracked_results.mat.

    if nargin < 2, error('Usage: wave_pipeline_unified(mode, inPath, outRoot[, saveImages])'); end
    if any(strcmpi(mode, {'verify','selftest'}))
        if nargin < 3 || isempty(outRoot), outRoot = inPath; end   % verify needs only an output root
    elseif nargin < 3
        error('Usage: wave_pipeline_unified(mode, inPath, outRoot[, saveImages])');
    end
    if nargin < 4 || isempty(saveImages), saveImages = true; end
    prm = defaultParams();
    prm.saveImages = saveImages;   % false -> numbers only, skip every figure/PNG (fast)
    rng(prm.seed);                 % determinism is already guaranteed; this only pins plot jitter

    if ~exist(outRoot,'dir'), mkdir(outRoot); end
    trackedRoot = fullfile(outRoot, '_tracked'); % tracker overlay PNGs only

    printVerifyBanner(prm);                       % remind user of key conventions

    rows       = {};      % one packed result struct per processed (subject,side)
    allResults = struct();% binary tracks saved for later

    switch lower(mode)
        case {'tree','run'}
            % Walk the dataset root: every <age> directory
            ageList = listDirs(inPath);
            for ia = 1:numel(ageList)
                ageName = ageList{ia};
                agePath = fullfile(inPath, ageName);
                eyeList = listDirs(agePath);            % every <eye> directory
                for ie = 1:numel(eyeList)
                    eyeName = eyeList{ie};
                    eyePath = fullfile(agePath, eyeName);
                    subs    = listDirs(eyePath);        % every <subject> directory
                    fprintf('\n===== %s | %s : %d subjects =====\n', ageName, eyeName, numel(subs));
                    for si = 1:numel(subs)
                        subName = subs{si};
                        subPath = fullfile(eyePath, subName);
                        % A subject folder with no frames of the expected type is skipped.
                        if isempty(dir(fullfile(subPath, ['*' prm.tail]))), continue; end
                        [rows, allResults] = runSubjectBothSides(subPath, ageName, eyeName, subName, ...
                            trackedRoot, outRoot, prm, rows, allResults);
                    end
                end
            end

        case 'one'
            % Single subject: infer age/eye/subject from the last 3 path parts
            [ageName, eyeName, subName] = inferMeta(inPath);
            fprintf('\n===== single subject: %s | %s | %s =====\n', ageName, eyeName, subName);
            [rows, allResults] = runSubjectBothSides(inPath, ageName, eyeName, subName, ...
                trackedRoot, outRoot, prm, rows, allResults);

        case {'verify','selftest'}
            % Synthetic ground-truth verification. Self-contained: builds
            % synthetic sequences of known velocity and runs them through the
            % same detection code used on the real data.
            verifyMode(outRoot, prm);
            return;

        otherwise
            error('mode must be ''tree'', ''one'', or ''verify''.');
    end

    % Aggregate outputs (only if anything was processed)
    if ~isempty(rows)
        writeMasterCSV(fullfile(outRoot,'all_results.csv'), rows);
        if prm.saveImages, saveMasterSummary(fullfile(outRoot,'master_summary.png'), rows); end    
    end
    save(fullfile(outRoot,'all_tracked_results.mat'), 'allResults');
    fprintf('\n  Done: %d sequences -> %s\n\n', numel(rows), outRoot);
end


% ══════════════════════════════════════════════════════════════════════
function prm = defaultParams()
% Single source of truth for every tunable. Fields shared by the
% segmentation, warp, tracker and profile branches live here so the
% branches cannot drift.

    % ── Labelling / device convention (VERIFY before trusting outputs) ─
    % rightEyeImageRightIs : For a RIGHT-eye image, what is the image-RIGHT
    %                       side anatomically? 'temporal' (the default) or
    %                       'nasal'. WRONG here => every nasal/temporal
    %                       label inverts silently.
    prm.rightEyeImageRightIs = 'temporal';

    % ── Frame reading + crop (shared by ALL stages) ───────────────────
    % cropRect   : [x y w h] pixel rectangle inside each JPG to keep.
    %              The cornea should sit inside this box.
    %              Warped output width rectW = w+1.
    % startFrame : first JPG index to read (1-based, sorted by name).
    % endFrame   : last  JPG index to read.
    % tail       : filename suffix / extension to look for.
    % mfs        : median-filter size for edge segmentation (px window).
    prm.cropRect   = [370, 30, 139, 129];    % [x y w h]; warped width rectW = w+1 = 140
    prm.startFrame = 50;   prm.endFrame = 100;
    prm.tail       = '.JPG';
    prm.mfs        = 11;                      % median-filter size for edge segmentation

    % ── Geometry construction ────────────────────────────────────────
    % aSmooth : span of smooth() applied to anterior/posterior surfaces
    %           before computing the midline arc. Larger = smoother arc.
    prm.aSmooth    = 11;

    % ── Temporal repair of the shared segmentation ────────────────────
    % The cornea deforms smoothly across frames, so a frame whose cornea
    % surfaces jump away from their neighbours is a segmentation error.
    % A short median in the frame direction repairs spikes and fills
    % whole-frame failures WITHOUT flattening the real deformation trend.
    %
    % geomTimeMed  : odd number of frames for the temporal median filter.
    %                Set to 1 to disable spike repair.
    % geomFillGaps : true -> linearly interpolate along time the frames
    %                that failed segmentation entirely.
    prm.geomTimeMed  = 3;                     % odd # of frames for the temporal median
    prm.geomFillGaps = true;                  % interpolate frames that failed outright

    % ── Strict-binary construction (used by the binary branch) ────────
    % strictNBINS      : number of intensity bins used to quantise the
    %                    positive frame-to-frame differences. A frame's
    %                    difference image is binned into NBINS levels and
    %                    bins above a noise-derived threshold are kept.
    % strictSE_LENGTH  : length (px) of the line structuring element used
    %                    to close gaps in the binary mask (morphology).
    % strictSE_ANGLE   : orientation (deg) of that line element. 60 deg
    %                    is slanted to match the typical wave front.
    % strictMIN_AREA   : smallest blob area (px) kept after area filtering.
    % strictMAX_JUMP    : max centroid jump (px) allowed between
    %                    consecutive frames for a blob to be kept.
    % strictNOISE_MULT : multiplier applied to the median positive-diff
    %                    value to obtain the raw intensity threshold.
    prm.strictNBINS      = 32;
    prm.strictSE_LENGTH  = 15;
    prm.strictSE_ANGLE   = 60;
    prm.strictMIN_AREA   = 20;
    prm.strictMAX_JUMP   = 10;                % px continuity gate between frames
    prm.strictNOISE_MULT = 3;

    % ── Profile sampling / detection ──────────────────────────────────
    % nDepth     : number of perpendicular samples stacked to form one
    %              depth-averaged 1-D profile.
    % trimFrac   : fraction of the half-thickness trimmed at each side
    %              when stacking. 0.10 keeps the central 80% of the
    %              corneal thickness.
    % tangSign  : sign of the tangential direction convention used when
    %              tilting profile lines (+1 or -1).
    % arcSmooth : Gaussian sigma (in arc units) used to smooth each
    %              raw 1-D profile before detrending (sudden fluctuations/trends).
    % phiBank   : list of obliquity angles (deg) tested when scanning for
    %              the wave's slant.
    % phiValid  : [min max] range of acceptable phi values; angles
    %              outside this range are not considered.
    % sStep     : arc-length spacing (arc px) at which the geometry is
    %              resampled for the profile branch.00
    % smoothMid : span of the smoothing applied when building the
    %              midline (anterior/posterior average) before arc length.
    % smoothThk : span of the moving median applied to per-column
    %              corneal thickness before resampling onto the arc.
    % detrendWin: width (px) of the moving-median baseline removed from
    %              each profile; reveals the wave's bump above the floor.
    % localWin  : half-width (px) of the local-maximum window used for
    %              peak detection.
    % snrStrong : SNR a peak must exceed to be a "strong" detection.
    % snrWeak   : SNR a peak must exceed to be a "weak" but usable one.
    % maxCand   : maximum number of candidate peaks kept per frame.
    prm.nDepth     = 20;    prm.trimFrac  = 0.10;
    prm.tangSign   = -1;    prm.arcSmooth = 1.5;
    prm.phiBank    = -40:2.5:40;
    prm.phiValid   = [10, 40];
    prm.sStep      = 1.0;                     % arc sampling step (arc unit)
    prm.smoothMid  = 11;    prm.smoothThk = 15;
    prm.detrendWin = 31;    prm.localWin  = 5;
    prm.snrStrong  = 4.0;   prm.snrWeak   = 2.5;  prm.maxCand = 4;

    % Asymmetric edge guard: the wave EXITS on the right, so keep the
    % left guard (baseline needs half a detrend window there) but shrink
    % the right guard so detections near the exit edge are not masked.
    % edgeGuardL : px trimmed from the LEFT  end of the valid profile.
    % edgeGuardR : px trimmed from the RIGHT end of the valid profile.
    prm.edgeGuardL = 20;    prm.edgeGuardR = 8;

    % ── Trajectory (RANSAC + refinement) ──────────────────────────────
    % vMin, vMax : accepted wave speed range (arc px per frame). Candidate
    %              lines outside this range are rejected by RANSAC.
    % ransacTol  : max residual (arc px) for a point to count as an inlier.
    % minInliers : minimum number of inliers needed to accept a RANSAC line.
    % seedFrames : number of earliest inliers used to seed the initial
    %              trajectory slope.
    % searchWin  : half-width (arc px) of the local search window used
    %              during per-frame refinement.
    % reacquireR : forward-only "re-acquire" radius. If the in-window peak
    %              is weak but a STRONG peak sits within this radius, the
    %              tracker snaps to it; this lets the tracker recover.
    % maxCoast   : max number of consecutive predicted-only frames before
    %              the tracker stops (prevents an endless coasting tail).
    % velPts     : number of recent accepted points used to estimate the
    %              short-term slope for prediction. Small so it follows
    %              deceleration.
    % orphanGap  : a fused detection with no neighbour within this many
    %              frames is dropped (treated as a noise blip/minor event).
    prm.vMin = 1.0;  prm.vMax = 8.0;
    prm.ransacTol = 4.0;  prm.minInliers = 5;
    prm.seedFrames = 5;                       % earliest inliers fix the initial slope
    prm.searchWin  = 15;
    prm.reacquireR = 20;                      % strong-peak re-acquire reach (forward)
    prm.maxCoast   = 4;                       % predicted-only frames before stopping
    prm.velPts     = 3;                       % short prediction window (follows deceleration)
    prm.orphanGap  = 3;                       % lone detections with no neighbour are dropped

    % ── Fusion ────────────────────────────────────────────────────────
    % agreeTol         : max arc-px difference between the binary and the
    %                    profile position (after bias correction) for the
    %                    two measurements to be considered "agreeing".
    % minBothForDelta  : minimum number of frames where BOTH branches
    %                    detected something before the bias Delta is
    %                    estimated. Below this, Delta is set to 0.
    prm.agreeTol = 5.0;
    prm.minBothForDelta = 4;

    % ── Physical scale (optional) ─────────────────────────────────────
    % mmPerPx    : millimetres per pixel (corneal arc). Used to convert
    %              fused speed to m/s. Leave NaN to skip the conversion.
    % msPerFrame : milliseconds per frame. Same as above.
    prm.mmPerPx = 8.5 / 576;   % = 0.014756944 mm/px
    prm.msPerFrame = 1000 / 4330;% = 0.230946882 ms/frame

    % ── Output controls ───────────────────────────────────────────────
    % dispMedian   : display-only median-filter size for the eye-test
    %                backdrop. Detection is run on the RAW image, not
    %                this denoised copy. Set to 0 to show raw images.
    % savePerFrame : true -> save one diagnostic figure per frame.
    % saveContact  : true -> save a contact sheet (all frames on a grid).
    prm.dispMedian   = 3;                     % DISPLAY-ONLY denoise for the eye-test backdrop
    prm.savePerFrame = true;                  % per-frame profile eye test
    prm.saveContact  = false;                  % contact sheet
    prm.seed           = 1;        % RNG seed (numeric path is deterministic regardless)
    prm.saveImages     = true;     % overwritten by the 4th argument above
    prm.minValidFrames = 5;        % QC: fewest accepted profile frames for a usable slope
end


% ══════════════════════════════════════════════════════════════════════
function printVerifyBanner(prm)
% Print a startup banner reminding the user of the conventions that,
% if wrong, silently flip labels. Read it every run.
    fprintf('\n  ============================================================\n');
    fprintf('   VERIFY: RIGHT eye, image-RIGHT treated as %s\n', upper(prm.rightEyeImageRightIs));
    fprintf('           (wrong => every nasal/temporal label inverts SILENTLY)\n');
    fprintf('   single shared segmentation + temporal median (%d fr)\n', prm.geomTimeMed);
    fprintf('   binary tracked in memory, frame-aligned\n');
    fprintf('  ============================================================\n\n');
end


% ══════════════════════════════════════════════════════════════════════
function [ageName, eyeName, subName] = inferMeta(subPath)
% Infer <age>/<eye>/<subject> from the last three components of a subject path.
    p     = regexprep(subPath, '[\\/]+$', '');      % strip trailing slashes
    parts = regexp(p, '[\\/]+', 'split');           % split on any slash
    parts = parts(~cellfun('isempty', parts));      % drop empties
    assert(numel(parts) >= 3, ...
        'Cannot infer age/eye/subject from "%s"; expected ...\\<age>\\<eye>\\<subject>.', subPath);
    subName = parts{end};                            % last  = subject
    eyeName = parts{end-1};                          % 2nd-last = eye
    ageName = parts{end-2};                          % 3rd-last = age
end


% ══════════════════════════════════════════════════════════════════════
function [rows, allResults] = runSubjectBothSides(subPath, ageName, eyeName, subName, ...
        trackedRoot, outRoot, prm, rows, allResults)
% Run the full chain for both image sides of one subject.
%   sideIdx 1 -> imgSide 'R', flipLR=false  (image already right-facing)
%   sideIdx 2 -> imgSide 'L', flipLR=true   (mirror so it looks like a right eye)

    for sideIdx = 1:2
        if sideIdx==1, imgSide='R'; flipLR=false;
        else,          imgSide='L'; flipLR=true;  end

        % Translate the image side into an anatomical meridian label
        % (nasal or temporal) using the convention set in prm.
        meridian  = nasalTemporal(eyeName, imgSide, prm.rightEyeImageRightIs);
        % Legacy short tag combining eye + image side (used for the tracker output folder)
        oldSide   = oldSideName(eyeName, imgSide);
        % Unique folder name for this (subject, image side, meridian)
        tag       = sprintf('%s_%s_%s', subName, imgSide, meridian);
        outFolder = fullfile(outRoot, ageName, eyeName, tag);
        trackOut  = fullfile(trackedRoot, ageName, eyeName, oldSide, subName);
        % Pack the metadata so downstream functions can stamp it on outputs.
        meta = struct('age',ageName,'eye',eyeName,'subject',subName, ...
                      'imgSide',imgSide,'meridian',meridian,'oldSide',oldSide);

        fprintf('  [%s|%s|%s|%s->%s]\n', ageName, eyeName, subName, imgSide, meridian);

        try
            [R, binTrack] = runOneSide(subPath, flipLR, outFolder, trackOut, meta, prm);
            rows{end+1} = R; %#ok<AGROW>
            if ~isempty(binTrack)
                % Make a valid MATLAB field name and stash the binary track.
                key = matlab.lang.makeValidName([ageName '_' eyeName '_' oldSide '_' subName]);
                bt  = binTrack;
                bt.ageGroup = ageName; bt.eyeSide = eyeName;
                bt.sideName = oldSide; bt.subject = subName;
                allResults.(key) = bt;
            end
        catch ME
            fprintf('      FAILED: %s\n', ME.message);
        end
    end
end


% ══════════════════════════════════════════════════════════════════════
function [R, binTrack] = runOneSide(subPath, flipLR, outFolder, trackOut, meta, prm)
% The unified per-(subject,side) chain. Everything downstream of the raw
% frames flows from ONE read and ONE segmentation.

    s2 = prm.cropRect(3) + 1;                 % cropped/warped width (=140)

    % ── (1) READ ONCE ─────────────────────────────────────────────────
    % Returns a cell array of cropped grayscale frames I{1..nF}.
    [I, nF] = readFrames(subPath, flipLR, prm);

    % ── (2) SEGMENT ONCE per frame ────────────────────────────────────
    % AntStack, PosStack : nF x s2 matrices of anterior / posterior surface
    %                      row indices per column. NaN row on failure.
    [AntStack, PosStack] = segmentAllFrames(I, prm);

    % ── (3) TEMPORAL SMOOTH the shared surfaces ───────────────────────
    % Spike repair and whole-frame gap filling along the time direction.
    [AntStack, PosStack, geomFixed] = temporalSmoothSurfaces(AntStack, PosStack, prm); %#ok<ASGLU>

    % ── (4) ONE GEOMETRY per frame, shared by both branches ───────────
    % G{t} is a struct carrying arc-length coordinates and the resampled
    % midline / normals used by the warp and the profile branch.
    G = cell(nF,1);
    for t = 1:nF
        if all(~isfinite(AntStack(t,:))), G{t} = []; continue; end
        G{t} = buildGeometry(AntStack(t,:), PosStack(t,:), prm);
    end

    % Per-sequence warp height = median corneal thickness across the run.
    thkAll = PosStack - AntStack;
    rect_H = round(median(thkAll(:), 'omitnan'));
    if ~isfinite(rect_H) || rect_H < 5
        % Degenerate thickness means the segmentation never locked on;
        % skip this side cleanly and write an empty result.
        warning('Degenerate thickness for %s|%s side %s; skipping.', meta.age, meta.subject, meta.imgSide);
        R = packResult(meta, nF, false(nF,1), false(nF,1), emptyF(nF), NaN, '', prm);
        binTrack = []; return;
    end

    % ── (5a) BINARY branch: warp + strict binary, all in memory ───────
    warped = cell(nF,1);
    for t = 1:nF
        if isempty(G{t}), warped{t} = zeros(rect_H, s2);
        else,             warped{t} = warpFrame(I{t}, G{t}, rect_H, prm); end
    end
    bgStack = buildBGStack(warped);                    % normalised backdrops
    strict  = strictBinaryFromWarped(warped, prm);    % strict{1} empty, strict{t} for t>=2

    % ── (5b) TRACK the binary in memory (frame-aligned) ───────────────
    binTrack = trackComponentInMem(strict, bgStack, trackOut, prm.saveImages);

    % ── (6) PROFILE branch: sigma0 + best phi, then detect + fuse ─────
    [sigma0, phiMed] = phiScan(I, G, prm);
    R = profileAndFuse(I, G, sigma0, phiMed, binTrack, outFolder, meta, prm);
end


% ══════════════════════════════════════════════════════════════════════
function [A, P] = segmentAllFrames(I, prm)
% Run the per-frame segmenter on every frame ONCE.
% A frame whose segmentation throws (e.g. fewer than 3 Canny regions) is
% left as a NaN row; the temporal repair in the next stage fills it from
% its neighbours.
    nF = numel(I);
    s2 = prm.cropRect(3) + 1;
    A = NaN(nF, s2);  P = NaN(nF, s2);
    for t = 1:nF
        try
            [a, p] = segmentA(I{t}, prm);
            A(t,:) = a;  P(t,:) = p;
        catch ME
            fprintf('      frame %d: segmentation failed (%s) -> temporal repair will fill it\n', ...
                t, ME.message);
        end
    end
end


% ══════════════════════════════════════════════════════════════════════
function [A, P, geomFixed] = temporalSmoothSurfaces(A, P, prm)
% Repair the shared anterior/posterior surfaces along TIME, once, for
% both branches. Two operations, both column-by-column and NaN-aware:
%   (a) a short temporal median (width prm.geomTimeMed) removes
%       single-frame spikes while preserving the slow deformation trend;
%   (b) whole frames that failed segmentation are filled by temporal
%       interpolation from the frames that did segment.
% geomFixed flags frames that were empty on input but are populated on output (useful for diagnostics).
    [nF, W]   = size(A);
    geomFixed = false(nF,1);
    have      = any(isfinite(A), 2);          % which frames had ANY valid data
    if nnz(have) < 3, return; end             % too little to repair against (nnz=number of non-zero elements)      

    % (a) temporal median spike-repair, column by column
    if prm.geomTimeMed > 1
        r  = (prm.geomTimeMed - 1) / 2;
        Af = A;  Pf = P;
        for t = 1:nF
            lo = max(1, t - r);  hi = min(nF, t + r);
            Af(t,:) = median(A(lo:hi, :), 1, 'omitnan');
            Pf(t,:) = median(P(lo:hi, :), 1, 'omitnan');
        end
        A = Af;  P = Pf;
    end

    % (b) fill whole-missing frames per column (linear interpolation in time)
    if prm.geomFillGaps
        for c = 1:W
            va = isfinite(A(:,c));
            if nnz(va) >= 2, A(:,c) = interp1(find(va), A(va,c), (1:nF)', 'linear', NaN); end
            vp = isfinite(P(:,c));
            if nnz(vp) >= 2, P(:,c) = interp1(find(vp), P(vp,c), (1:nF)', 'linear', NaN); end
        end
    end

    geomFixed = ~have & any(isfinite(A), 2);
end


% ══════════════════════════════════════════════════════════════════════
function G = buildGeometry(Ant, Pos, prm)
% Build ONE arc-length geometry from the (temporally-smoothed) anterior
% and posterior surfaces. The same arc is used by the warp branch and
% resampled for the profile branch, so both branches share a single
% coordinate system: the binary-to-arc conversion later is exact.
%
% The struct carries two coordinate views:
%   - PROFILE fields (s, xm, ym, tx, ty, nx, ny, Tn, colValid) sampled at sStep.
%   - WARP fields    (AsCol, PsCol, arcCol, nxCol, nyCol, totalArc) per column.
% Both are derived from the SAME smoothed midline, so totalArc is the
% same value the fusion later uses.

    s2 = numel(Ant);                % number of source columns

    % columns the segmenter committed a boundary to (it fills, so usually all)
    colValid0 = isfinite(Ant) & isfinite(Pos) & (Pos > Ant);
    Ant = fillmissing(Ant, 'linear', 'EndValues', 'nearest');
    Pos = fillmissing(Pos, 'linear', 'EndValues', 'nearest');

    % Smooth anterior and posterior separately, then average for the midline.
    As = reshape(smooth(Ant, prm.aSmooth), 1, s2);
    Ps = reshape(smooth(Pos, prm.aSmooth), 1, s2);
    As = fillmissing(As, 'linear', 'EndValues', 'nearest');
    Ps = fillmissing(Ps, 'linear', 'EndValues', 'nearest');
    mid = (As + Ps) / 2;

    % Arc length along the midline, per source column.
    dMid = gradient(mid);
    arc  = [0, cumsum(sqrt(1 + dMid(1:end-1).^2))];
    total_arc = arc(end);

    % Per-column unit normal pointing toward increasing row (posterior).
    dlen  = sqrt(1 + dMid.^2);
    nxCol = -dMid ./ dlen;
    nyCol =  1    ./ dlen;
    thkCol = Ps - As;

    % ── PROFILE view: resample the same arc at sStep ──────────────────
    x   = 1:s2;
    sU  = 0:prm.sStep:total_arc;
    xm  = interp1(arc, x,    sU, 'linear', 'extrap');   % source col per arc point
    ym  = interp1(arc, mid,  sU, 'linear', 'extrap');   % midline row per arc point
    dmv = interp1(arc, dMid, sU, 'linear', 'extrap');
    thv = interp1(arc, movmedian(thkCol, prm.smoothThk), sU, 'linear', 'extrap');
    cv  = interp1(x, double(colValid0), xm, 'nearest', 0) > 0.5;
    L = sqrt(1 + dmv.^2);
    tx = 1 ./ L;  ty = dmv ./ L;  nx = -dmv ./ L;  ny = 1 ./ L;  Tn = thv .* tx;

    G = struct('s',sU,'xm',xm,'ym',ym,'tx',tx,'ty',ty,'nx',nx,'ny',ny, ...
               'Tn',Tn,'colValid',cv,'H',prm.cropRect(4)+1,'W',s2, ...
               'AntCol',Ant,'PosCol',Pos, ...
               'AsCol',As,'PsCol',Ps,'arcCol',arc,'nxCol',nxCol,'nyCol',nyCol, ...
               'totalArc',total_arc);
end


% ══════════════════════════════════════════════════════════════════════
function out = warpFrame(img, G, rect_H, prm) %#ok<INUSD>
% Unbend one frame onto an arc-uniform rectangle of size rect_H x rectW.
%
% Output column j sits at arc (j-1)/(rectW-1)*totalArc. Within a column,
% we walk the corneal normal from the anterior surface (t=0) to the
% posterior surface (t=1), sampling bilinearly from the source image.
%
% INPUTS
%   img    : the source grayscale frame (Hs x Ws).
%   G      : geometry struct from buildGeometry (carries the arc and normals).
%   rect_H : output rectangle height (px). Typically the median thickness.
%   prm    : parameter struct (only cropRect is used here, to size the width).
%
% OUTPUT
%   out    : rect_H x rectW warped grayscale image. Background outside the
%            cornea is left as 0. A 3x3 median filter cleans up speckle.
    s2    = G.W;
    rectW = s2;
    [Hs, Ws] = size(img);

    arc_out = linspace(0, G.totalArc, rectW);
    src_col = interp1(G.arcCol, 1:s2, arc_out, 'linear', 'extrap');

    out = zeros(rect_H, rectW);
    for j = 1:rectW
        xc  = src_col(j);
        ac  = interp1(1:s2, G.AsCol, xc, 'linear', 'extrap');   % anterior row
        pc  = interp1(1:s2, G.PsCol, xc, 'linear', 'extrap');   % posterior row
        nxc = interp1(1:s2, G.nxCol, xc, 'linear', 'extrap');
        nyc = interp1(1:s2, G.nyCol, xc, 'linear', 'extrap');
        if ~isfinite(ac) || ~isfinite(pc), continue; end
        th = pc - ac;
        if th <= 1, continue; end
        for y_out = 1:rect_H
            t_frac = (y_out - 1) / (rect_H - 1);
            src_x  = xc + nxc * t_frac * th;
            src_y  = ac + nyc * t_frac * th;
            if src_x < 1 || src_x > Ws || src_y < 1 || src_y > Hs, continue; end
            % Bilinear interpolation from the source image.
            x_lo = floor(src_x); x_hi = min(x_lo+1, Ws);
            y_lo = floor(src_y); y_hi = min(y_lo+1, Hs);
            wx = src_x - x_lo;   wy = src_y - y_lo;
            out(y_out, j) = ...
                (1-wy)*(1-wx)*img(y_lo,x_lo) + (1-wy)*wx*img(y_lo,x_hi) + ...
                   wy*(1-wx)*img(y_hi,x_lo) +     wy*wx*img(y_hi,x_hi);
        end
    end
    out = medfilt2(out, [3 3]); % because salt and pepper noise is present
end


% ══════════════════════════════════════════════════════════════════════
function bg = buildBGStack(warped)
% Normalise each warped frame to a uint8 greyscale backdrop for the
% tracker overlay. Each frame is min/max-stretched over its non-zero
% (corneal) region; pixels that were 0 (background) are kept at 0.
%
% INPUT  : warped   : cell array of warped grayscale frames (any frame may be empty)
% OUTPUT : bg      : cell array of uint8 normalised backdrops of the same length
    nF = numel(warped);
    bg = cell(nF,1);
    for t = 1:nF
        fv = warped{t};
        if isempty(fv), bg{t} = []; continue; end
        m = (fv == 0);
        if all(m(:)), bg{t} = uint8(zeros(size(fv))); continue; end
        mn = min(fv(~m)); mx = max(fv(~m));
        fn = (fv - mn) / (mx - mn + eps);
        fn(m) = 0;
        bg{t} = uint8(fn * 255);
    end
end


% ══════════════════════════════════════════════════════════════════════
function strict = strictBinaryFromWarped(warped, prm)
% Build the strict binary "wave event" masks from the warped stack.
%
% Pipeline (per frame, after the first):
%   1. Take the POSITIVE frame-to-frame difference (wave = brightening).
%      Mask it where either frame is background.
%   2. Estimate a noise floor from the median positive-diff and a
%      global cap from the 99.5th percentile of all positive diffs.
%   3. Quantise the difference image into NBINS bins and threshold at the
%      noise-derived level.
%   4. Morphologically close with a slanted line element to bridge gaps.
%   5. Drop blobs smaller than strictMIN_AREA, keep the top 3 by area.
%   6. Cross-frame continuity gate: a blob is kept only if its centroid
%      is within strictMAX_JUMP of some blob in the previous frame.
%
% INPUT  : warped   : cell array of warped frames.
% OUTPUT : strict  : cell array of logical masks (same length).
%                    strict{1} is empty (needs a previous frame).
%                    strict{t} aligns with warped frame t.
    nF = numel(warped);

    % Find the common warp dimensions.
    rH = 0; rW = 0;
    for t = 1:nF
        if ~isempty(warped{t}), [rH, rW] = size(warped{t}); break; end
    end
    strict = cell(nF,1);
    for t = 1:nF, strict{t} = false(max(rH,1), max(rW,1)); end
    if rH == 0, return; end

    % Positive frame differences, masked where either frame is background.
    all_diffs = zeros(rH, rW, nF);
    for t = 2:nF
        if isempty(warped{t}) || isempty(warped{t-1}), continue; end
        d = warped{t} - warped{t-1};
        d(d < 0) = 0;
        d((warped{t} == 0) | (warped{t-1} == 0)) = 0;
        all_diffs(:,:,t) = d;
    end

    nz = all_diffs(all_diffs > 0);
    if isempty(nz), return; end

    noise_floor = median(nz);
    raw_thresh  = noise_floor * prm.strictNOISE_MULT;
    global_max  = prctile(nz, 99.5);

    NBINS = prm.strictNBINS;
    edges = linspace(0, global_max, NBINS + 1);
    bin_thresh = find(edges >= raw_thresh, 1) - 1;
    if isempty(bin_thresh) || bin_thresh < 1, bin_thresh = 1;       end
    if bin_thresh >= NBINS,                    bin_thresh = NBINS-1; end
    se_thin = strel('line', prm.strictSE_LENGTH, prm.strictSE_ANGLE);

    % FIRST PASS — top-3 candidate blobs per frame.
    frame_cents = cell(nF,1);
    frame_pix   = cell(nF,1);
    for t = 2:nF
        d_capped = min(all_diffs(:,:,t), global_max);
        lvl = discretize(d_capped, edges);
        lvl(d_capped >= global_max) = NBINS;
        bw_q      = lvl >= bin_thresh;
        bw_closed = imclose(bw_q, se_thin);

        CC_cl = bwconncomp(bw_closed);
        ar_cl = cellfun(@numel, CC_cl.PixelIdxList);
        bw_clean = false(rH, rW);
        for k = 1:CC_cl.NumObjects
            if ar_cl(k) >= prm.strictMIN_AREA, bw_clean(CC_cl.PixelIdxList{k}) = true; end
        end

        CC2   = bwconncomp(bw_clean);
        st2   = regionprops(CC2, 'Area', 'Centroid');
        areas = [st2.Area];
        cents = zeros(0,2); pix = {};
        if CC2.NumObjects > 0
            [~, idx2] = sort(areas, 'descend');
            keep = idx2(1:min(3, end));           % top 3 blobs by area
            for k = 1:numel(keep)
                cents(end+1,:) = st2(keep(k)).Centroid;      %#ok<AGROW>
                pix{end+1}     = CC2.PixelIdxList{keep(k)};   %#ok<AGROW>
            end
        end
        frame_cents{t} = cents;
        frame_pix{t}   = pix;
    end

    % SECOND PASS — keep only blobs continuous with the previous frame.
    for t = 2:nF
        curr = frame_cents{t};
        pxs  = frame_pix{t};
        if t > 2, prev = frame_cents{t-1}; else, prev = []; end
        bs = false(rH, rW);
        for k = 1:size(curr,1)
            c = curr(k,:);
            keep_this = isempty(prev) || ...
                        min(sqrt(sum((prev - c).^2, 2))) <= prm.strictMAX_JUMP;
            if keep_this, bs(pxs{k}) = true; end
        end
        strict{t} = bs;
    end
end


% ══════════════════════════════════════════════════════════════════════
function [sigma0, phiMed] = phiScan(I, G, prm)
% Estimate the profile noise sigma0 for each frame and scan the wave's
% obliquity angle phi. The sequence-level phi is the median of the
% per-frame best phi over frames with a STRONG response. Amplitude is
% normalised by the FIXED sigma0 so the score is blur-proof.
%
% OUTPUTS
%   sigma0 : nF x 1 noise sigma per frame (NaN where undetermined).
%   phiMed : scalar median phi (deg) used for the whole sequence.
    nF    = numel(I);
    nPhi  = numel(prm.phiBank);
    okPhi = prm.phiBank >= prm.phiValid(1) & prm.phiBank <= prm.phiValid(2);

    sigma0 = NaN(nF,1); phiBest = NaN(nF,1); strong = false(nF,1);
    for t = 1:nF
        if isempty(G{t}), continue; end
        [P0,V0]  = sampleProfile(I{t}, G{t}, 0, prm);
        [Pd0,M0] = prepProfile(P0, V0, prm);
        s0 = robustSigma(Pd0(M0));
        if ~isfinite(s0) || s0 <= 0, continue; end
        sigma0(t) = s0;
        sc = -inf(1,nPhi);
        for ip = 1:nPhi
            if ~okPhi(ip), continue; end
            [P,V]  = sampleProfile(I{t}, G{t}, prm.phiBank(ip), prm);
            [Pd,M] = prepProfile(P, V, prm);
            [~,amp] = peakMetrics(Pd, M, prm);
            sc(ip) = amp / s0;
        end
        [mx, ib] = max(sc);
        if isfinite(mx), phiBest(t) = prm.phiBank(ib); strong(t) = mx >= prm.snrStrong; end
    end
    if any(strong), phiMed = median(phiBest(strong), 'omitnan');
    else,           phiMed = mean(prm.phiValid); end
end


% ══════════════════════════════════════════════════════════════════════
function R = profileAndFuse(I, G, sigma0, phiMed, binTrack, outFolder, meta, prm)
% Profile detection + RANSAC trajectory + fusion with the binary track,
% then all the diagnostics.
%
% The binary-to-arc conversion is EXACT and FRAME-ALIGNED: xb(t) is the
% tracker centroid at profile frame t, and totalArc(t) is that frame's
% own arc length from the shared geometry.
    if prm.saveImages
        if ~exist(outFolder,'dir'), mkdir(outFolder); end
    end
    nF = numel(I);

    % 1D profiles + candidate peaks for every frame.
    [Pdet, Msk, cand, totalArc] = buildProfiles(I, G, sigma0, phiMed, nF, prm);

    % RANSAC over all candidate peaks -> seed line + inlier set.
    [line0, inl] = ransacTrajectory(cand, prm);
    if isempty(line0)
        sP = NaN(nF,1); snrP = NaN(nF,1); okP = false(nF,1); predS = NaN(nF,1);
    else
        % Refine the trajectory frame-by-frame using a short prediction
        % window, with re-acquire and coast-limit guards.
        [sP, snrP, okP, predS] = refineTrajectory(Pdet, Msk, sigma0, line0, inl, prm);
    end

    % ── binary track -> arc coordinates (exact, no offset) ────────────
    sB = NaN(nF,1); okB = false(nF,1);
    rectW = prm.cropRect(3) + 1;
    if ~isempty(binTrack) && isfield(binTrack,'centroid')
        xb = binTrack.centroid(:,1);          % length nF, aligned to profile frames
        for t = 1:nF
            if t > numel(xb) || isnan(xb(t)) || isempty(G{t}) || ~isfinite(totalArc(t))
                continue;
            end
            % The warp resampled the arc uniformly onto rectW columns, and
            % totalArc(t) is that same arc, so warped x maps to arc exactly:
            sB(t)  = (xb(t)-1)/(rectW-1) * totalArc(t);
            okB(t) = true;
        end
    end

    % Combine profile + binary into a single fused trajectory.
    F = fuseTracks(sP, okP, sB, okB, prm);
    if prm.saveImages
        % Diagnostics.
        saveFusionDiag(fullfile(outFolder,'fusion_summary.png'), sP, okP, sB, okB, F, meta, prm, phiMed);
        if prm.savePerFrame
            saveAllFrames(outFolder, I, G, Pdet, Msk, cand, sigma0, ...
                          sP, snrP, okP, okB, predS, F, meta, prm);
        end
        if prm.saveContact
            saveContactSheet(fullfile(outFolder,'frames_contact.png'), I, G, F, meta, prm);
        end
    end
    % Pack a compact result row for the CSV / master summary.
    R = packResult(meta, nF, okP, okB, F, phiMed, '', prm);
    save(fullfile(outFolder,'result.mat'), 'R','sP','snrP','okP','sB','okB','F','phiMed','predS');

    fprintf('      prof %2d | bin %2d | agree %2d (%.0f%%) | interp %2d | delta %+.1f | v %.2f | grade %s\n', ...
        sum(okP), sum(okB), F.nAgree, 100*F.agreeRate, F.nInterp, F.delta, F.vProf, R.grade);
end


% ══════════════════════════════════════════════════════════════════════
function F = emptyF(nF)
% Minimal placeholder fusion struct for sequences skipped before fusion,
% so packResult and the CSV writer have the fields they read.
    F = struct('delta',NaN,'sBc',NaN(nF,1),'resid',NaN(nF,1), ...
               'status',{repmat({'absent'},nF,1)}, ...
               'statusFinal',{repmat({'absent'},nF,1)}, ...
               'sFuse',NaN(nF,1),'sFinal',NaN(nF,1),'fit',[NaN NaN], ...
               'nBoth',0,'nAgree',0,'nConflict',0,'nInterp',0,'agreeRate',0, ...
               'vProf',NaN,'vBin',NaN,'R2',NaN,'span',[NaN NaN]);
end


% ══════════════════════════════════════════════════════════════════════
% SEGMENTATION  (per frame, anterior/posterior surface extraction)
% ══════════════════════════════════════════════════════════════════════
function [Anterior, Posterior] = segmentA(img, prm)
% Per-frame anterior/posterior surface extraction. Returns row vectors
% (1 x s2). The nchoosek thickness repair, Canny top-3 selection and
% fillmissing are used. Errors (e.g. a frame with <3 edge regions)
% propagate to the caller, which records a NaN row that the temporal
% repair then fills from neighbouring frames.
%
% INPUTS
%   img : grayscale frame (HxW).
%   prm : parameter struct (uses prm.mfs for the median filter size).
%
% OUTPUTS
%   Anterior  : row of anterior surface row-indices per column (NaN where missing).
%   Posterior : row of posterior surface row-indices per column (NaN where missing).
    mfs = prm.mfs;
    [s1, s2] = size(img); %#ok<ASGLU>
    Anterior  = NaN(1, s2);
    Posterior = NaN(1, s2);

    % Light median denoise then Canny edges.
    J  = medfilt2(img, [mfs mfs]);
    BW = edge(J, 'Canny');

    % Keep only the 3 largest Canny regions (approx cornea + 2 lobes).
    Properties = regionprops(BW, 'Area');
    A = zeros(1, length(Properties));
    for m = 1:length(Properties)
        A(m) = Properties(m).Area;
    end
    [~, inde]  = sort(A, 'descend');
    thresh     = A(inde(3)) + 1;
    BW_3       = bwareaopen(BW, thresh);

    % Per column, the topmost foreground pixel is anterior, the
    % bottommost is posterior.
    for m = 1:s2
        Profil = BW_3(:,m);
        ind    = find(Profil == 1);
        if ~isempty(ind)
            Anterior(m)  = ind(1);
            Posterior(m) = ind(end);
        end
    end

    % First thickness sanity check: drop columns where the local
    % thickness is less than half the median thickness.
    thickness        = Posterior - Anterior;
    thickness_median = median(thickness, 'omitnan');

    for m = 1:s2
        col_thickness = Posterior(m) - Anterior(m);
        if ~isfinite(col_thickness) || col_thickness < thickness_median * 0.5
            Anterior(m)  = NaN;
            Posterior(m) = NaN;
        end
    end

    Anterior  = fillmissing(Anterior,  'linear');
    Posterior = fillmissing(Posterior, 'linear');
    thickness          = Posterior - Anterior;
    thickness_median   = median(thickness, 'omitnan');

    % If the median thickness is implausible, try pairs of the top 3
    % regions and keep the pair whose median thickness is in range.
    if thickness_median > 27 || thickness_median < 22

        CC = bwconncomp(BW);
        component_areas = cellfun(@numel, CC.PixelIdxList);
        [~, sorted_idx] = sort(component_areas, 'descend');
        top3_idx = sorted_idx(1:min(3, numel(sorted_idx)));

        pairs          = nchoosek(1:length(top3_idx), 2);
        best_thickness = 0;
        best_Anterior  = Anterior;
        best_Posterior = Posterior;

        for pp = 1:size(pairs,1)
            BW_pair = false(size(BW));
            BW_pair(CC.PixelIdxList{top3_idx(pairs(pp,1))}) = true;
            BW_pair(CC.PixelIdxList{top3_idx(pairs(pp,2))}) = true;

            Ant_try  = NaN(1, s2);
            Post_try = NaN(1, s2);
            for m = 1:s2
                Profil = BW_pair(:,m);
                ind    = find(Profil == 1);
                if ~isempty(ind)
                    Ant_try(m)  = ind(1);
                    Post_try(m) = ind(end);
                end
            end

            th_med = median(Post_try - Ant_try, 'omitnan');
            if th_med < 27 && th_med > best_thickness
                best_thickness = th_med;
                best_Anterior  = Ant_try;
                best_Posterior = Post_try;
            end
        end

        Anterior  = best_Anterior;
        Posterior = best_Posterior;

        % Remove spikes in the anterior that deviate too far from a
        % moving-median reference; for each bad column re-pick the
        % nearest in-range foreground pixel.
        ant_ref   = movmedian(Anterior, 21, 'omitnan');
        deviation = abs(Anterior - ant_ref);
        bad_cols  = deviation > thickness_median * 0.4;

        for m = find(bad_cols)
            Profil = BW_pair(:,m);
            ind    = find(Profil == 1);
            if ~isempty(ind)
                valid_ind = ind(abs(ind - ant_ref(m)) <= thickness_median * 0.4);
                if ~isempty(valid_ind)
                    Anterior(m)  = valid_ind(1);
                    Posterior(m) = valid_ind(end);
                else
                    Anterior(m)  = NaN;
                    Posterior(m) = NaN;
                end
            end
        end

        Anterior  = fillmissing(Anterior,  'linear');
        Posterior = fillmissing(Posterior, 'linear');

        thickness = Posterior - Anterior;
        too_thin  = thickness < 10;
        Posterior(too_thin) = Anterior(too_thin) + 20;
    end

    Anterior  = fillmissing(Anterior,  'nearest');
    Posterior = fillmissing(Posterior, 'nearest');
end


% ══════════════════════════════════════════════════════════════════════
% IN-MEMORY BINARY TRACKER
% Operates on the in-memory strict masks and backdrops, producing a
% frame-aligned trajectory.
% ══════════════════════════════════════════════════════════════════════
function tracked = trackComponentInMem(bwStack, bgStack, outputFolder, saveImages)
% Track a single retinal wave blob through the binary mask stack.
%
% The function works in two stages:
%
%   STAGE 1 — INIT SCAN (frames 1 ... INIT_SCAN_FRAMES)
%     Reads every blob in the first INIT_SCAN_FRAMES frames, then
%     evaluates each blob as a candidate "seed" by simulating forward
%     tracking from it. The best seed (lowest composite score) becomes
%     initCentroid.
%
%   STAGE 2 — MAIN TRACKING LOOP (frames 1 ... nFrames)
%     Uses initCentroid to latch onto the wave blob and then follows it
%     frame-by-frame using a predicted position + penalty scoring system.
%
% INPUTS
%   bwStack       : cell array of logical masks (one per frame).
%   bgStack       : cell array of uint8 greyscale backdrops (one per frame
%                   may be empty). Used only for the overlay visualisation.
%   outputFolder  : where overlay PNGs are written (one per input frame).
%
% OUTPUT
%   tracked.centroid  [nFrames x 2]  NaN where no detection; x in col-1,
%                                    fixedY (image midline) in col-2.
%   tracked.area      [nFrames x 1]  blob area in pixels; NaN on misses.
%   tracked.frames    [nFrames x 1]  frame index vector 1..nFrames.
%   tracked.fixedY    scalar         fixed row used for Y coordinate.

    % ── Tunable parameters ─────────────────────────────────────────────
    %
    % MAX_DISPLACEMENT (15 px)
    %   Hard radius for accepting a blob as the tracked target in the
    %   main loop. A match is only accepted if the nearest blob is
    %   within this distance AND the composite score < SCORE_THRESHOLD.
    %
    % FIXED_DRIFT [2, 0] px/frame
    %   Expected per-frame displacement of the wave. Used both as a
    %   predictor in the scan phase and in the main tracking loop.
    %   X=+2 encodes a rightward-propagating wave; Y=0 because the
    %   cornea has been "unbent" so the wave is approximately horizontal.
    %
    % MAX_MISSING (15 frames)
    %   How many consecutive missed frames are tolerated before the
    %   tracker gives up and stops advancing lastCentroid.
    %
    % INIT_SCAN_FRAMES (60)
    %   How many frames are pre-read for the seed selection. If the
    %   wave enters the ROI after frame 60, the scan will fail and all
    %   three retry attempts will also fail.
    %
    % AREA_ALPHA (0.15)
    %   Exponential-smoothing weight for the running area estimate.
    %   Low value = slow adaptation (area changes must be gradual).
    %
    % SCORE_THRESHOLD (30)
    %   Maximum allowable composite score in the main loop. The score
    %   is distance + direction_penalty + area_penalty; a blob is only
    %   accepted if BOTH distance <= MAX_DISPLACEMENT AND score < 30.
    %
    % MAX_RETRIES (3)
    %   Number of times the init-scan is repeated with progressively
    %   relaxed constraints if no valid seed is found.
    %
    % RETRY_AREA_WIDEN (2.0)
    %   On each retry the area acceptance window is multiplied by this
    %   factor (high end x2, low end /2).
    %
    % RETRY_DX_RELAX (10 px)
    %   On each retry the leftward-motion penalty threshold is loosened
    %   by this many pixels (attempt 1: -5, attempt 2: -15, attempt 3: -25).
    %
    % RETRY_SCORE_DECAY (0.5)
    %   On each retry the match/streak rewards are halved, so the score
    %   is numerically smaller and a weaker tracklet can win.
    %
    % MIN_MATCHES (3)
    %   A seed candidate must successfully match at least this many
    %   subsequent frames in the scan window to be eligible.
    %
    % DRIFT_X_REWARD (3.0)
    %   Score bonus (per pixel of net rightward displacement) granted to
    %   a seed whose simulated tracklet ends further right than it started.
    % -----------------------------------------------------------------------
    MAX_DISPLACEMENT  = 15;
    FIXED_DRIFT       = [2, 0];
    MAX_MISSING       = 15;
    INIT_SCAN_FRAMES  = 60;
    AREA_ALPHA        = 0.3;
    SCORE_THRESHOLD   = 30;
    MAX_RETRIES       = 3;
    RETRY_AREA_WIDEN  = 2.0;
    RETRY_DX_RELAX    = 10;
    RETRY_SCORE_DECAY = 0.5;
    MIN_MATCHES       = 5;
    DRIFT_X_REWARD    = 3.0;

    if saveImages && ~exist(outputFolder,'dir'), mkdir(outputFolder); end

    % Frames arrive as in-memory stacks (binary + backdrop), one cell
    % per profile frame, so the tracker is INDEX-ALIGNED with the
    % profile branch: tracked.centroid(t,:) corresponds to profile
    % frame t (no file offset).
    nFrames = numel(bwStack);
    assert(nFrames > 0, 'Empty binary stack passed to trackComponentInMem.');

    % Pre-allocate output logs with NaN so missing detections are explicit.
    centroid_log  = NaN(nFrames, 2);
    area_log      = NaN(nFrames, 1);

    % ── Fixed Y coordinate ─────────────────────────────────────────────
    % The cornea has been straightened, so the wave runs approximately
    % along a horizontal strip. We fix the reported Y position to the
    % vertical midline of the image rather than using the raw centroid Y,
    % to keep vertical noise from contaminating the x(t) velocity metric.
    fixedY = size(bwStack{1}, 1) / 2;
    fprintf('  Fixed Y (image midline): %.1f\n\n', fixedY);

    % ══════════════════════════════════════════════════════════════════
    % STAGE 1A — PRE-READ SCAN WINDOW
    % Read blobs from the first INIT_SCAN_FRAMES frames once and cache
    % them. This avoids re-reading images on every retry attempt.
    % ══════════════════════════════════════════════════════════════════
    scan_limit     = min(INIT_SCAN_FRAMES, nFrames);
    scan_centroids = cell(scan_limit, 1);   % {frame} -> [nBlobs x 2] centroids
    scan_areas     = cell(scan_limit, 1);   % {frame} -> [nBlobs x 1] areas

    fprintf('  Scanning %d frames for persistent component...\n', scan_limit);
    for si = 1:scan_limit
        bw = logical(bwStack{si});   % already a binary event mask
        [cens, ars] = getComponentsFast(bw);
        if ~isempty(cens)
            scan_centroids{si} = cens;
            scan_areas{si}     = ars;
        end
        % Frames with no blobs remain as empty cells (handled downstream).
    end

    % Initialise outputs of the seed-selection block.
    initCentroid = [];
    initArea     = [];
    initFrame    = 1;

    % ══════════════════════════════════════════════════════════════════
    % STAGE 1B — SEED SELECTION WITH UP TO MAX_RETRIES ATTEMPTS
    %
    % HOW THE SEED IS CHOSEN — step-by-step
    % ──────────────────────────────────────
    % For every blob ci in every scan frame si, the algorithm runs a
    % mini forward-tracking simulation:
    %
    %   a) Start at blob ci's centroid (startPos) with its area.
    %   b) For each subsequent frame fj in the scan window:
    %        • Predict the next position: curPos + FIXED_DRIFT.
    %        • Find the closest blob in frame fj to that prediction.
    %        • Reject blobs that moved too far LEFT (dx < dxThreshold).
    %        • Add a distance penalty (+50) to blobs whose area differs
    %          too much from the running area estimate (areaHigh/areaLow).
    %        • If the best match is within MAX_DISPLACEMENT+10 px -> count
    %          it as a "match", update curPos and the running area.
    %        • Otherwise -> miss; reset the consecutive-match counter.
    %
    %   c) After simulating all forward frames, compute a SCORE:
    %
    %        score = meanDist
    %                - matches    x matchReward      (more matches = better)
    %                - maxConsec  x consecReward     (long unbroken runs = better)
    %                - netDx      x DRIFT_X_REWARD   (rightward travel = better)
    %
    %      Lower score = better candidate. The blob with the lowest
    %      score across all (si, ci) pairs becomes the initial seed.
    % ══════════════════════════════════════════════════════════════════
    for attempt = 1:MAX_RETRIES

        % Each retry halves the reward weights so that a weaker
        % tracklet can achieve a sufficiently negative score to win.
        decayFactor  = RETRY_SCORE_DECAY ^ (attempt - 1);
        matchReward  = 10 * decayFactor;    % per matched frame
        consecReward = 15 * decayFactor;    % per frame in best consecutive run

        % Area acceptance window widens on each retry.
        areaMultiplier = RETRY_AREA_WIDEN ^ (attempt - 1);
        areaHigh = 4    * areaMultiplier;   % max area_ratio allowed without penalty
        areaLow  = 0.25 / areaMultiplier;   % min area_ratio allowed without penalty

        % Leftward-motion rejection threshold (px).
        % Attempt 1: -5 px  (strict: only 5 px of leftward wobble allowed)
        % Attempt 2: -15 px (relaxed)
        % Attempt 3: -25 px (very relaxed)
        dxThreshold = -5 - RETRY_DX_RELAX * (attempt - 1);

        if attempt == 1
            fprintf('  Attempt %d: area=[%.3f %.1f]  dx_thresh=%.0f  score_weights=[%.1f %.1f]\n', ...
                attempt, areaLow, areaHigh, dxThreshold, matchReward, consecReward);
        else
            fprintf('  Retry %d/%d: area=[%.3f %.1f]  dx_thresh=%.0f  score_weights=[%.1f %.1f]\n', ...
                attempt, MAX_RETRIES, areaLow, areaHigh, dxThreshold, matchReward, consecReward);
        end

        bestScore    = inf;
        initCentroid = [];
        initArea     = [];
        initFrame    = 1;

        % ── Outer loop: every starting frame in the scan window ────────
        for si = 1:scan_limit
            if isempty(scan_centroids{si}), continue; end

            % ── Inner loop: every blob in starting frame si ────────────
            for ci = 1:size(scan_centroids{si}, 1)

                % Remember where this candidate blob started.
                % initCentroid is recorded at position (si, ci), NOT at
                % the end of the simulation, so the main loop anchors to
                % the blob's actual position in its first appearance frame.
                startPos = scan_centroids{si}(ci,:);
                curPos   = startPos;      % current simulated tracker position
                curArea  = scan_areas{si}(ci);

                % Counters for this candidate's tracklet simulation.
                matches   = 0;    % total frames where a blob was matched
                totalDist = 0;    % sum of matched distances (for mean later)
                consec    = 0;    % current consecutive-match streak length
                maxConsec = 0;    % longest consecutive streak seen so far

                % ── Forward simulation ─────────────────────────────────
                for fj = si+1:scan_limit
                    if isempty(scan_centroids{fj})
                        % No blobs detected in frame fj -> count as a miss,
                        % break the consecutive streak but keep curPos.
                        consec = 0;
                        continue;
                    end

                    cands    = scan_centroids{fj};
                    areas_fj = scan_areas{fj}(:);
                    areas_fj(areas_fj < 1) = 1;   % guard against zero area

                    % Predict next position using the fixed drift model.
                    pred  = curPos + FIXED_DRIFT;
                    dists = sqrt(sum((cands - pred).^2, 2));   % [nBlobs x 1]

                    % Penalise blobs that moved more than dxThreshold px
                    % leftward from the current position.
                    dx = cands(:,1) - curPos(1);
                    dists(dx < dxThreshold) = inf;

                    % Soft area filter: blobs outside the expected area
                    % range get +50 added to their effective distance.
                    % This makes them unlikely to win the nearest-blob
                    % contest but does not hard-exclude them.
                    area_ratio = areas_fj / curArea;
                    dists(area_ratio > areaHigh | area_ratio < areaLow) = ...
                        dists(area_ratio > areaHigh | area_ratio < areaLow) + 50;

                    [minD, bestK] = min(dists);

                    % Accept the match only if within the relaxed threshold.
                    if minD <= MAX_DISPLACEMENT + 10
                        matches   = matches   + 1;
                        consec    = consec    + 1;
                        maxConsec = max(maxConsec, consec);
                        totalDist = totalDist + minD;
                        % Advance simulated position to the matched blob.
                        curPos  = cands(bestK,:);
                        curArea = (1-AREA_ALPHA)*curArea + AREA_ALPHA*areas_fj(bestK);
                    else
                        consec = 0;   % miss: reset streak but keep curPos
                    end
                end
                % ── End of forward simulation for this seed ────────────

                % Quality gate: discard seeds that matched too few frames.
                if matches < MIN_MATCHES, continue; end

                meanDist = totalDist / matches;

                % Net rightward displacement reward. A large positive netDx
                % makes the score very negative (= very good).
                netDx       = curPos(1) - startPos(1);
                driftReward = netDx * DRIFT_X_REWARD;

                % Composite score — lower is better:
                %   + meanDist       penalises seeds that tracked far-away blobs
                %   - matches x reward rewards persistence
                %   - maxConsec x reward rewards unbroken runs
                %   - driftReward    rewards rightward net motion
                score = meanDist ...
                        - matches   * matchReward  ...
                        - maxConsec * consecReward ...
                        - driftReward;

                if score < bestScore
                    bestScore    = score;
                    initFrame    = si;                      % first-appearance frame
                    initCentroid = scan_centroids{si}(ci,:); % position at that frame
                    initArea     = scan_areas{si}(ci);       % area at that frame
                end
            end % ci blob loop
        end % si frame loop

        if ~isempty(initCentroid)
            if attempt > 1
                fprintf('  Component found on attempt %d.\n', attempt);
            end
            break;   % successful seed found — no need for further retries
        end
        fprintf('  No persistent component found on attempt %d.\n', attempt);
    end % retry loop

    % If all retries failed, return an all-NaN result rather than crashing.
    if isempty(initCentroid)
        warning('No persistent component found after %d attempts. Skipping.', MAX_RETRIES);
        tracked = struct('centroid', centroid_log, 'area', area_log, ...
                         'frames', (1:nFrames)', 'fixedY', fixedY);
        return;
    end

    fprintf('  Selected init at frame %d @ [%.1f %.1f] area=%.0f\n\n', ...
        initFrame, initCentroid(1), initCentroid(2), initArea);

    % ══════════════════════════════════════════════════════════════════
    % STAGE 2 — MAIN TRACKING LOOP
    %
    % Processes every frame 1..nFrames in order. Two sub-modes:
    %
    %  LATCH MODE (lastCentroid is empty, frame >= initFrame)
    %    Used only once, at the very first detection after initFrame.
    %    Searches for a blob near the linearly-extrapolated initCentroid
    %    position using a generous 3x MAX_DISPLACEMENT radius and a
    %    looser area check (0.33 - 3x). Once a blob is found, the
    %    tracker switches permanently to TRACK MODE.
    %
    %  TRACK MODE (lastCentroid is set)
    %    Each frame, predict next position = lastCentroid + FIXED_DRIFT.
    %    Score all candidate blobs in the frame:
    %      score = distance + direction_penalty + area_penalty
    %    Accept the best blob only if BOTH:
    %      • distance <= MAX_DISPLACEMENT (15 px)
    %      • score    < SCORE_THRESHOLD    (30)
    %    On a miss, increment missingStreak and advance lastCentroid by
    %    FIXED_DRIFT (ghost prediction) so the predictor stays roughly
    %    aligned with where the wave should be. After MAX_MISSING (15)
    %    consecutive misses, lastCentroid is frozen (no further advance).
    % ══════════════════════════════════════════════════════════════════
    lastCentroid  = [];        % empty until the first successful latch
    smoothArea    = initArea;  % exponentially-smoothed area reference
    missingStreak = 0;         % consecutive missed frames counter

    for i = 1:nFrames

        bw = logical(bwStack{i});
        [binH, binW] = size(bw);

        % Backdrop for the overlay: the greyscale BG frame if present.
        if ~isempty(bgStack) && numel(bgStack) >= i && ~isempty(bgStack{i})
            bgImg = bgStack{i};
            if size(bgImg,1)~=binH || size(bgImg,2)~=binW
                bgImg = imresize(bgImg, [binH binW]);
            end
        else
            bgImg = uint8(bw) * 255;   % fallback: the binary frame itself
        end

        [centroids, areas] = getComponentsFast(bw);

        foundTarget = false;
        idx         = [];

        if ~isempty(centroids)
            areas(areas < 1) = 1;   % guard against zero area

            % ── LATCH MODE ─────────────────────────────────────────────
            if isempty(lastCentroid) && i >= initFrame

                % Linearly extrapolate from initCentroid to frame i using
                % the fixed drift model as a starting estimate.
                targetPred      = initCentroid + FIXED_DRIFT * (i - initFrame);
                d               = sqrt(sum((centroids - targetPred).^2, 2));
                area_ratio_init = areas / initArea;

                % Exclude blobs with area < 1/3 or > 3x the seed area.
                d(area_ratio_init > 3.0 | area_ratio_init < 0.33) = inf;

                [minD, idx] = min(d);

                % Generous 3x radius for the first latch because the
                % extrapolation error grows with (i - initFrame).
                if minD <= MAX_DISPLACEMENT * 3
                    smoothArea  = initArea;
                    foundTarget = true;
                end

            % ── TRACK MODE ─────────────────────────────────────────────
            elseif ~isempty(lastCentroid) && missingStreak < MAX_MISSING

                % Predict next position from the last confirmed centroid.
                predicted = lastCentroid + FIXED_DRIFT;

                % Euclidean distance from each candidate blob to the prediction.
                dists = sqrt(sum((centroids - predicted).^2, 2));

                % DIRECTION PENALTY: blobs moving leftward relative to the
                % last confirmed position are penalised.
                %   dx = +10 px (rightward) -> penalty = 0
                %   dx =   0 px (stationary) -> penalty = 0
                %   dx =  -5 px (leftward)   -> penalty = 40
                %   dx = -10 px              -> penalty = 80
                dx          = centroids(:,1) - lastCentroid(1);
                dir_penalty = 40 * max(0, -dx / 5);

                % AREA PENALTY: |log2(area_ratio)| penalises blobs that
                % have grown or shrunk unexpectedly. log2 makes the
                % penalty symmetric (halving and doubling give equal
                % penalties).
                area_ratio   = areas / smoothArea;
                area_penalty = 8 * abs(log2(area_ratio));

                score = dists + dir_penalty + area_penalty;

                [best_score, best_idx] = min(score);
                best_dist              = dists(best_idx);

                % DUAL GATE: both conditions must be satisfied for acceptance.
                if best_dist <= MAX_DISPLACEMENT && best_score < SCORE_THRESHOLD
                    foundTarget = true;
                    idx         = best_idx;
                    % Update the running area estimate via EMA.
                    smoothArea  = (1-AREA_ALPHA)*smoothArea + AREA_ALPHA*areas(idx);

                    fprintf('Frame %3d: [%5.1f %5.1f] area=%4.0f (exp=%4.0f) dist=%.1f score=%.1f\n', ...
                        i, centroids(idx,1), fixedY, ...
                        areas(idx), smoothArea, best_dist, best_score);
                else
                    fprintf('Frame %3d: miss  dist=%.1f score=%.1f (thresh=%d)\n', ...
                        i, best_dist, best_score, SCORE_THRESHOLD);
                end
            end
        end % ~isempty(centroids)

        % ── Record or advance ──────────────────────────────────────────
        if foundTarget
            centroid_log(i, 1) = centroids(idx, 1);
            centroid_log(i, 2) = fixedY;          % fixed Y, not raw centroid Y
            area_log(i)        = areas(idx);
            lastCentroid       = [centroids(idx,1), fixedY];
            missingStreak      = 0;
        else
            missingStreak = missingStreak + 1;
            % Ghost-advance the predictor so that after a short gap the
            % tracker can still re-latch near the correct position. After
            % MAX_MISSING consecutive misses the wave is assumed gone;
            % ghost advancement stops to avoid drifting to the image edge.
            if ~isempty(lastCentroid) && missingStreak < MAX_MISSING
                lastCentroid = lastCentroid + FIXED_DRIFT;
            end
        end
        if saveImages
        % Write the visualisation overlay for this frame.
        overlayImg = renderOverlayFast(bgImg, bw, centroids, idx, foundTarget, centroid_log(1:i,:));
        imwrite(overlayImg, fullfile(outputFolder, sprintf('f%03d_tracked.png', i)));
        end
    end % main loop

    tracked          = struct();
    tracked.centroid = centroid_log;   % x positions (col 1), fixedY (col 2)
    tracked.area     = area_log;       % blob area per frame
    tracked.frames   = (1:nFrames)';   % frame index vector for downstream use
    tracked.fixedY   = fixedY;
end


% ══════════════════════════════════════════════════════════════════════


function [centroids, areas] = getComponentsFast(bw)
% Label connected components and return their centroids and pixel-count
% areas without calling regionprops (faster for many blobs).
%
% Uses 8-connectivity so diagonally-adjacent pixels are treated as one blob.
% Returns empty arrays when no blobs are present (safe for all callers).
%
% INPUT
%   bw : logical mask.
%
% OUTPUT
%   centroids  [nBlobs x 2]  [mean_col, mean_row] i.e. [x, y]
%   areas      [nBlobs x 1]  number of pixels per blob

    [L, num] = bwlabel(bw, 8);
    if num == 0
        centroids = [];
        areas     = [];
        return;
    end

    centroids = zeros(num, 2);
    areas     = zeros(num, 1);

    % Vectorised extraction: find all foreground pixel coordinates once,
    % then group by label. Faster than looping over bwlabel output.
    [rows, cols] = find(L);
    idxs         = L(sub2ind(size(L), rows, cols));   % label of each pixel

    for k = 1:num
        r_k = rows(idxs == k);
        c_k = cols(idxs == k);
        centroids(k,:) = [mean(c_k), mean(r_k)];   % [x=col, y=row]
        areas(k)       = numel(r_k);
    end
end


% ══════════════════════════════════════════════════════════════════════


function overlayImg = renderOverlayFast(imgRGB, bw, centroids, targetIdx, foundTarget, centroid_log)
% Produce an annotated overlay image for one frame.
%
% Composites four layers onto the greyscale (or RGB) background:
%   1. Orange semi-transparent tint on all detected binary blobs.
%   2. Jet-coloured trail showing the tracked x(t) path so far, where
%      colour encodes per-segment speed (blue=slow, red=fast).
%   3. Small 3x3 dot at each previously-detected centroid position.
%   4. White crosshair on the current frame's accepted target blob.
% A speed colourbar with min/max labels is appended at the bottom.
%
% INPUTS
%   imgRGB      : background image (uint8 grayscale or RGB).
%   bw          : binary mask of detected blobs.
%   centroids   : [nBlobs x 2] candidate blob centroids in [x y] form.
%   targetIdx   : index into centroids of the accepted target (or []).
%   foundTarget : true if a target was accepted this frame.
%   centroid_log: [t x 2] history of accepted centroids up to this frame.
%
% OUTPUT
%   overlayImg  : uint8 RGB annotated image with top/bottom padding.

    BLOB_OPACITY = 0.35;
    BLOB_COLOR   = [255, 180, 0];   % orange

    % Ensure RGB input for pixel-level colour manipulation.
    if size(imgRGB,3)==1
        imgRGB = cat(3, imgRGB, imgRGB, imgRGB);
    end
    overlayImg = uint8(imgRGB);

    % ── Layer 1: orange tint on all binary blobs ───────────────────────
    % Alpha-blend: new_pixel = (1-a)*original + a*BLOB_COLOR
    if ~isempty(bw) && any(bw(:))
        bwMask = logical(bw);
        for ch = 1:3
            plane         = double(overlayImg(:,:,ch));
            plane(bwMask) = plane(bwMask)*(1-BLOB_OPACITY) + BLOB_COLOR(ch)*BLOB_OPACITY;
            overlayImg(:,:,ch) = uint8(plane);
        end
    end

    [H0, W, ~] = size(overlayImg);

    % ── Padding ────────────────────────────────────────────────────────
    topPad    = round(0.10 * H0);
    bottomPad = round(0.35 * H0);
    overlayImg = cat(1, uint8(zeros(topPad, W, 3)), overlayImg, ...
                        uint8(zeros(bottomPad, W, 3)));
    H = H0 + topPad;   % row index of the original image's top edge after padding

    % ── Compute per-segment speed for colour mapping ───────────────────
    % Speed is distance between consecutive DETECTED points divided by
    % the frame gap, giving px/frame rather than px/step (gap-corrected).
    valid = find(~isnan(centroid_log(:,1)));
    nV    = numel(valid);

    segSpeed = zeros(max(nV-1, 1), 1);
    if nV >= 2
        pts       = centroid_log(valid, :);
        dx        = diff(pts(:,1));
        dy        = diff(pts(:,2));
        frameGaps = diff(valid);          % frames between consecutive detections
        segSpeed  = sqrt(dx.^2 + dy.^2) ./ frameGaps;   % px / frame
    end

    maxSpeed = max(segSpeed);
    if maxSpeed < eps, maxSpeed = 1; end   % avoid divide-by-zero

    % Assign each point the speed of the segment leading out of it; the
    % last point reuses the last segment's speed.
    pointSpeed = zeros(nV, 1);
    if nV >= 2
        pointSpeed(1:nV-1) = segSpeed;
        pointSpeed(nV)     = segSpeed(end);
    end

    % ── Layer 2: jet-coloured trail lines ─────────────────────────────
    if nV >= 2
        for k = 1:nV-1
            p1    = centroid_log(valid(k),   :);
            p2    = centroid_log(valid(k+1), :);
            t     = segSpeed(k) / maxSpeed;         % 0=slow(blue)...1=fast(red)
            col   = uint8(jet_color_local(t) * 255);
            n_pts = max(round(norm(p2-p1)), 1);     % number of interpolation steps
            xs    = round(linspace(p1(1), p2(1), n_pts));
            ys    = round(linspace(p1(2), p2(2), n_pts));
            for p = 1:n_pts
                if xs(p)>0 && xs(p)<=W && ys(p)>0 && ys(p)<=H
                    overlayImg(ys(p)+topPad, xs(p), :) = col;
                end
            end
        end
    end

    % ── Layer 3: history dots (3x3 neighbourhood) ─────────────────────
    for k = 1:nV
        t   = pointSpeed(k) / maxSpeed;
        col = uint8(jet_color_local(t) * 255);
        cx  = round(centroid_log(valid(k), 1));
        cy  = round(centroid_log(valid(k), 2));
        for ddx = -1:1
            for ddy = -1:1
                if cx+ddx>0 && cx+ddx<=W && cy+ddy>0 && cy+ddy<=H
                    overlayImg(cy+ddy+topPad, cx+ddx, :) = col;
                end
            end
        end
    end

    % ── Layer 4: white crosshair on current target ─────────────────────
    if foundTarget && ~isempty(targetIdx)
        cx = round(centroids(targetIdx, 1));
        cy = round(centroid_log(valid(end), 2));   % use last confirmed Y
        for d = -4:4
            if cy>0 && cy<=H && cx+d>0 && cx+d<=W
                overlayImg(cy+topPad, cx+d, :) = 255;
            end
            if cy+d>0 && cy+d<=H && cx>0 && cx<=W
                overlayImg(cy+d+topPad, cx, :) = 255;
            end
        end
    end

    % ── Speed colourbar ────────────────────────────────────────────────
    barLeft   = round(0.10 * W);
    barRight  = round(0.90 * W);
    barTop    = H + 25;
    barBottom = H + 30;
    barWidth  = barRight - barLeft + 1;

    for c = 1:barWidth
        t   = (c-1) / (barWidth-1);
        col = uint8(255 * jet_color_local(t));
        overlayImg(barTop:barBottom, c+barLeft-1, 1) = col(1);
        overlayImg(barTop:barBottom, c+barLeft-1, 2) = col(2);
        overlayImg(barTop:barBottom, c+barLeft-1, 3) = col(3);
    end

    % Speed labels below the colourbar.
    minLabelSpeed = 0;
    maxLabelSpeed = max(segSpeed);

    overlayImg = insertText(overlayImg, ...
        [round(barLeft/2),  H + round(0.23*H0)], ...
        sprintf('%.1f px/fr', minLabelSpeed), ...
        'FontSize', 8, 'BoxOpacity', 0.6, 'BoxColor', 'black', 'TextColor', 'white');

    overlayImg = insertText(overlayImg, ...
        [round(0.7*W),      H + round(0.23*H0)], ...
        sprintf('%.1f px/fr', maxLabelSpeed), ...
        'FontSize', 8, 'BoxOpacity', 0.6, 'BoxColor', 'black', 'TextColor', 'white');

    overlayImg = insertText(overlayImg, ...
        [round(0.40*W),     H + round(0.08*H0)], ...
        'Speed', ...
        'FontSize', 8, 'BoxOpacity', 0, 'BoxColor', 'black', 'TextColor', 'white');
end


% ══════════════════════════════════════════════════════════════════════


function rgb = jet_color_local(t)
% Return an [R G B] triple for the jet colormap at t in [0,1].
%   t = 0   -> blue
%   t = 0.5 -> green/yellow
%   t = 1   -> red
% Implemented locally to avoid a dependency on the MATLAB colormap functions.
    t = max(0, min(1, t));
    r = min(1, max(0, 1.5 - abs(4*t - 3)));
    g = min(1, max(0, 1.5 - abs(4*t - 2)));
    b = min(1, max(0, 1.5 - abs(4*t - 1)));
    rgb = [r g b];
end


% ══════════════════════════════════════════════════════════════════════
% PROFILE + FUSION LEAF FUNCTIONS
% ══════════════════════════════════════════════════════════════════════
function [I, nF] = readFrames(imgFolder, flipLR, prm)
% Read every JPG in the folder, keep only frames [startFrame endFrame],
% convert to grayscale, optionally mirror left-right, and crop to cropRect.
%
% OUTPUTS
%   I  : cell array of cropped grayscale double images (one per kept frame).
%   nF : number of frames actually read.
    files = dir(fullfile(imgFolder, ['*' prm.tail]));
    assert(~isempty(files), 'No %s in %s', prm.tail, imgFolder);
    I = {};
    for k = 1:numel(files)
        if k < prm.startFrame || k > prm.endFrame, continue; end
        im = double(imread(fullfile(files(k).folder, files(k).name)));
        if size(im,3)==3, im = rgb2gray(im/255)*255; end
        if flipLR, im = fliplr(im); end
        I{end+1} = imcrop(im, prm.cropRect); %#ok<AGROW>
    end
    nF = numel(I);
    assert(nF > 2, 'Not enough frames in %s', imgFolder);
end


function [P,V] = sampleProfile(img,G,phiDeg,prm)
% Sample one depth-averaged 1-D profile along the arc, at obliquity phi.
%
% INPUTS
%   img     : grayscale source image (HxW).
%   G       : geometry struct from buildGeometry.
%   phiDeg  : obliquity angle (deg) tilting the profile lines along the tangent.
%   prm     : parameter struct (uses nDepth, trimFrac, tangSign).
%
% OUTPUTS
%   P : 1 x nS profile values (NaN where invalid).
%   V : 1 x nS logical mask of valid (in-image & in-cornea) points.
    nS = numel(G.s);
    fr = linspace(-(0.5-prm.trimFrac),(0.5-prm.trimFrac),prm.nDepth);
    acc = zeros(1,nS); cnt = zeros(1,nS);
    for k = 1:prm.nDepth
        v = fr(k)*G.Tn;
        [px,py] = tiltedLinePoints(G,v,phiDeg,prm.tangSign);
        val = bilinear(img,px,py); ok = ~isnan(val);
        acc(ok) = acc(ok)+val(ok); cnt(ok) = cnt(ok)+1;
    end
    V = (cnt==prm.nDepth) & G.colValid;
    P = NaN(1,nS); P(V) = acc(V)./prm.nDepth;
end


function [Pd,M,Ps] = prepProfile(P,V,prm)
% Smooth, detrend, and gate a raw profile so the wave bump stands out.
%
% OUTPUTS
%   Pd : 1 x nS detrended profile (raw minus moving-median baseline).
%   M  : 1 x nS logical "useful" mask (valid + away from edges).
%   Ps : 1 x nS smoothed raw profile (pre-detrend), useful for plotting.
    n=numel(P); Pd=NaN(1,n); M=false(1,n); Ps=NaN(1,n);
    idx=find(V);
    if numel(idx) < prm.detrendWin+prm.edgeGuardL+prm.edgeGuardR, return; end
    i0=idx(1); i1=idx(end);
    seg=P(i0:i1);
    seg=fillmissing(seg,'linear','EndValues','nearest');
    seg=gaussSmooth1D(seg,prm.arcSmooth);
    base=movmedian(seg,prm.detrendWin,'omitnan');
    Ps(i0:i1)=seg; Pd(i0:i1)=seg-base;
    g0=i0+prm.edgeGuardL; g1=i1-prm.edgeGuardR;   % asymmetric edge guard
    if g1<=g0, return; end
    M(g0:g1)=V(g0:g1);
end


function [px,py] = tiltedLinePoints(G,v,phiDeg,tangSign)
% Compute the source-image pixel coordinates for one tilted profile line.
% The line is perpendicular to the arc, tilted by phiDeg along the tangent.
    tp = tangSign*tand(phiDeg);
    px = G.xm + v.*G.nx + (v.*tp).*G.tx;
    py = G.ym + v.*G.ny + (v.*tp).*G.ty;
end


function y = gaussSmooth1D(x,sigma)
% Gaussian-smooth a 1-D signal with the given sigma, normalising the
% edge regions where the kernel hangs off the ends.
    if sigma<=0, y=x; return; end
    r=max(1,ceil(3*sigma));
    k=exp(-((-r:r).^2)/(2*sigma^2)); k=k/sum(k);
    y=conv(x,k,'same'); nz=conv(ones(size(x)),k,'same'); y=y./nz;
end


function [pk,amp,fwhm] = peakMetrics(Pd,M,prm)
% Find the most prominent local peak in the detrended profile.
%
% OUTPUTS
%   pk   : arc index of the peak (NaN if none).
%   amp  : peak amplitude above the local median.
%   fwhm : full-width half-max of the peak (in arc px).
    pk=NaN; amp=NaN; fwhm=NaN;
    idx=find(M);
    if numel(idx)<2*prm.localWin+1, return; end
    v=Pd(idx); med=median(v);
    best=NaN; bestVal=-inf;
    for k=1:numel(idx)
        i=idx(k); lo=i-prm.localWin; hi=i+prm.localWin;
        if lo<idx(1)||hi>idx(end), continue; end
        if Pd(i)>=max(Pd(lo:hi)) && Pd(i)>bestVal, bestVal=Pd(i); best=i; end
    end
    if isnan(best), return; end
    pk=best; amp=Pd(best)-med;
    half=med+amp/2;
    L=best; while L>idx(1)   && Pd(L)>half, L=L-1; end
    R=best; while R<idx(end) && Pd(R)>half, R=R+1; end
    fwhm=R-L;
end


function s = robustSigma(v)
% Robust noise sigma estimate: 1.4826 * MAD (median absolute deviation),
% which equals the standard deviation for a Gaussian but is insensitive
% to outliers. Needs at least 5 finite samples.
    v=v(isfinite(v));
    if numel(v)<5, s=NaN; return; end
    s=1.4826*median(abs(v-median(v)))+eps;
end


function v = bilinear(img,x,y)
% Vectorised bilinear interpolation of the grayscale image at subpixel
% (x,y) coordinates. Out-of-bounds points return NaN.
    [H,W]=size(img); v=NaN(size(x));
    ok = x>=1 & x<=W & y>=1 & y<=H;
    xo=x(ok); yo=y(ok);
    x0=floor(xo); x1=min(x0+1,W); y0=floor(yo); y1=min(y0+1,H);
    wx=xo-x0; wy=yo-y0;
    i00=sub2ind([H W],y0,x0); i01=sub2ind([H W],y0,x1);
    i10=sub2ind([H W],y1,x0); i11=sub2ind([H W],y1,x1);
    v(ok)=(1-wy).*((1-wx).*img(i00)+wx.*img(i01)) + wy.*((1-wx).*img(i10)+wx.*img(i11));
end


function C = candidatePeaks(Pd, M, s0, prm)
% Find the strongest local maxima in a detrended profile.
%
% OUTPUT
%   C : nCand x 3 matrix of [arc_idx, amplitude, SNR] for each candidate,
%       sorted by decreasing SNR and filtered by snrWeak.
    C = zeros(0,3); idx = find(M);
    if numel(idx) < 2*prm.localWin+1, return; end
    med = median(Pd(idx),'omitnan'); pk = [];
    for k = 1:numel(idx)
        i = idx(k); lo=i-prm.localWin; hi=i+prm.localWin;
        if lo<idx(1)||hi>idx(end), continue; end
        if Pd(i) >= max(Pd(lo:hi)), pk(end+1)=i; end %#ok<AGROW>
    end
    if isempty(pk), return; end
    amp = Pd(pk)-med; snr = amp/s0;
    [~,ord] = sort(snr,'descend');
    keep = ord(1:min(prm.maxCand,numel(ord)));
    C = [pk(keep)', amp(keep)', snr(keep)'];
    C = C(C(:,3) >= prm.snrWeak, :);
end


function [Pdet, Msk, cand, totalArc] = buildProfiles(I, G, sigma0, phi, nF, prm)
% Build the per-frame detrended profiles, valid masks, candidate peaks,
% and per-frame total arc length used by the trajectory fitter.
%
% OUTPUTS
%   Pdet     : nF x nS detrended profiles.
%   Msk      : nF x nS logical masks of usable points.
%   cand     : nF cell array of candidate peaks (each [nCand x 3]).
%   totalArc : nF x 1 per-frame arc length (same as G{t}.s(end)).
    nS = 0;
    for t=1:nF, if ~isempty(G{t}), nS = max(nS, numel(G{t}.s)); end, end
    Pdet=NaN(nF,nS); Msk=false(nF,nS); cand=cell(nF,1); totalArc=NaN(nF,1);
    for t = 1:nF
        if isempty(G{t}) || ~isfinite(sigma0(t)), continue; end
        totalArc(t) = G{t}.s(end);
        [P,V]    = sampleProfile(I{t}, G{t}, phi, prm);
        [Pd,M,~] = prepProfile(P, V, prm);
        n = numel(Pd); Pdet(t,1:n)=Pd; Msk(t,1:n)=M;
        cand{t} = candidatePeaks(Pd, M, sigma0(t), prm);
    end
end


function [best, inl] = ransacTrajectory(cand, prm)
% 2-point RANSAC over all candidate peaks. For every pair of candidate
% peaks on different frames, fit a straight line s(t) = v*t + b, keep
% the line that satisfies vMin <= v <= vMax and has the highest sum of
% inlier SNRs (residual within ransacTol).
%
% The seed velocity is taken from the EARLIEST few inliers only, so the
% seed reflects the wave where it is cleanest and fastest; the
% refinement adapts from there.
%
% OUTPUTS
%   best : struct with fields v (slope), b (intercept), f0 (seed frame),
%          s0 (seed arc position).
%   inl  : struct with fields frames (1 x k) and s (1 x k).
    best=[]; inl=struct('frames',[],'s',[]);
    pts = zeros(0,3);
    for t = 1:numel(cand)
        if isempty(cand{t}), continue; end
        pts = [pts; repmat(t,size(cand{t},1),1), cand{t}(:,1), cand{t}(:,3)]; %#ok<AGROW>
    end
    if size(pts,1) < prm.minInliers, return; end
    bestScore = -inf;
    for a = 1:size(pts,1)-1
        for b = a+1:size(pts,1)
            dt = pts(b,1)-pts(a,1);
            if dt == 0, continue; end
            v = (pts(b,2)-pts(a,2))/dt;
            if v < prm.vMin || v > prm.vMax, continue; end
            res = abs(pts(:,2) - (pts(a,2)+v*(pts(:,1)-pts(a,1))));
            in  = res <= prm.ransacTol;
            if sum(in) < prm.minInliers, continue; end
            fr = unique(pts(in,1))'; sc=0; fs=[]; ss=[];
            for ff = fr
                m = in & pts(:,1)==ff; rows = find(m);
                [~,kk] = min(res(rows)); r = rows(kk);
                sc = sc + pts(r,3);
                fs(end+1)=ff; ss(end+1)=pts(r,2); %#ok<AGROW>
            end
            if sc > bestScore
                bestScore = sc;
                [fs2, ord] = sort(fs(:)); ss2 = ss(ord)'; ss2 = ss2(:);
                nSeed = min(prm.seedFrames, numel(fs2));
                pf = polyfit(fs2(1:nSeed), ss2(1:nSeed), 1);
                best = struct('v',pf(1),'b',pf(2), 'f0',fs2(1),'s0',ss2(1));
                inl  = struct('frames',fs(:),'s',ss(:));
            end
        end
    end
end


function [trackS, trackSNR, trackOK, predS] = refineTrajectory(Pdet, Msk, sigma0, line0, inl, prm)
% Frame-by-frame trajectory refinement using a short prediction window,
% a forward-only search window, a re-acquire radius for strong peaks,
% and a coast limit that prevents endless prediction-only tails.
%
% INPUTS
%   Pdet   : nF x nS detrended profiles.
%   Msk    : nF x nS valid-point masks.
%   sigma0 : nF x 1 per-frame noise sigma.
%   line0  : RANSAC seed line (struct from ransacTrajectory).
%   inl    : RANSAC inlier set.
%   prm    : parameter struct.
%
% OUTPUTS
%   trackS   : nF x 1 accepted arc position per frame (NaN where missed).
%   trackSNR : nF x 1 SNR of the accepted peak (NaN where missed).
%   trackOK  : nF x 1 logical mask of accepted frames.
%   predS    : nF x 1 predicted arc position per frame (for diagnostics).
%
% Guards against the "death spiral" where a runaway prediction leaves the
% real peak behind:
%   (a) RE-ACQUIRE: if the in-window peak is weak/absent but a STRONG peak
%       exists within +-reacquireR, snap to it.
%   (b) COAST LIMIT: stop after maxCoast consecutive predicted-only frames.
%   (c) FORWARD ONLY: never step behind the last accepted position, and
%       predict from ACCEPTED measurements only.
    nF = size(Pdet,1);
    trackS=NaN(nF,1); trackSNR=NaN(nF,1); trackOK=false(nF,1); predS=NaN(nF,1);
    f0 = min(inl.frames); f1 = max(inl.frames);
    histF=[]; histS=[]; coast=0;

    nS = size(Pdet,2);   % arc extent; predictions beyond this are meaningless

    for t = f0:nF
        % ── predict from recent ACCEPTED history only ─────────────────
        % SHORT window (velPts, default 3) so the prediction follows
        % deceleration rather than lagging behind an outdated slope.
        if numel(histF) >= 2
            k  = max(1, numel(histF)-prm.velPts+1);
            pf = polyfit(histF(k:end)', histS(k:end)', 1);
            v  = pf(1);
            v  = min(max(v, 0), prm.vMax);          % never predict backward
            pred = histS(end) + v*(t - histF(end));
        elseif ~isempty(histF)
            pred = histS(end) + line0.v*(t - histF(end));
        else
            % First frame: anchor on the seed's OWN start, not a global
            % intercept extrapolated to frame 1.
            pred = line0.s0 + line0.v*(t - line0.f0);
        end

        % A prediction outside the arc is a runaway, not a position.
        % Clamp for the search; treat "ran off the end" as a stop once
        % we are past the corroborated span.
        if ~isempty(histS), pred = max(pred, histS(end)); end   % monotone predict
        if pred > nS
            if t > f1, break; end
            pred = nS;
        end
        predS(t) = pred;

        pd = Pdet(t,:); m = Msk(t,:);
        if ~any(m) || ~isfinite(sigma0(t)), coast=coast+1; if past(t,f1,coast,prm), break; end; continue; end
        med = median(pd(m),'omitnan');

        % Lower bound for ANY acceptance this frame: never behind history.
        floorS = 1; if ~isempty(histS), floorS = histS(end); end

        % ── (a) primary: best local max INSIDE the window ─────────────
        lo = max(floorS, round(pred - prm.searchWin));
        hi = min(nS,     round(pred + prm.searchWin));
        s_win = NaN; snr_win = -inf;
        if hi > lo
            w = m; w(1:lo-1)=false; w(hi+1:end)=false;
            if any(w)
                ii = find(w); [pv,pidx] = max(pd(ii)); s_win = ii(pidx);
                snr_win = (pv - med)/sigma0(t);
            end
        end

        % ── (b) re-acquire: STRONG peak just AHEAD of the window ──────
        % Forward only: rlo starts at the last accepted position, so a
        % re-acquire can never pull the track backward.
        s_re = NaN; snr_re = -inf;
        rlo = max(floorS, round(pred - prm.reacquireR));
        rhi = min(nS,     round(pred + prm.reacquireR));
        if rhi > rlo
            wr = m; wr(1:rlo-1)=false; wr(rhi+1:end)=false;
            if any(wr)
                ir = find(wr); [pvr,pir] = max(pd(ir)); s_re = ir(pir);
                snr_re = (pvr - med)/sigma0(t);
            end
        end

        % ── decide ────────────────────────────────────────────────────
        accepted = false;
        if snr_win >= prm.snrWeak
            s_hat = s_win; snr = snr_win; accepted = true;
        elseif snr_re >= prm.snrStrong && isfinite(s_re)
            s_hat = s_re;  snr = snr_re; accepted = true;
        end

        % Final monotonicity guard (belt and braces).
        if accepted && ~isempty(histS) && s_hat < histS(end)
            accepted = false;
        end

        if accepted
            trackS(t)=s_hat; trackSNR(t)=snr; trackOK(t)=true;
            histF(end+1)=t; histS(end+1)=s_hat; %#ok<AGROW>
            coast = 0;
        else
            coast = coast + 1;
            if past(t,f1,coast,prm), break; end
        end
    end
end


function stop = past(t, f1, coast, prm)
% Return true once we are beyond the corroborated span AND have coasted
% too long without an accepted detection.
    stop = (t > f1) && (coast >= prm.maxCoast);
end


function F = fuseTracks(sP, okP, sB, okB, prm)
% Fuse the profile track (wave centre) with the binary track (wave leading
% edge). The binary blob sits AHEAD of the profile centre by roughly half
% the wave width; we estimate that systematic offset (Delta) robustly and
% subtract it before comparing the two.
%
% INPUTS
%   sP  : nF x 1 profile arc positions.
%   okP : nF x 1 logical: profile detection present.
%   sB  : nF x 1 binary  arc positions.
%   okB : nF x 1 logical: binary  detection present.
%   prm : parameter struct (uses agreeTol, minBothForDelta, orphanGap).
%
% The reported velocity is the slope of the PROFILE track (vProf). The binary
% track is used only to verify that both methods followed the same feature,
% through the agreement count and the offset Delta. The combined (sFuse/sFinal)
% trajectory is retained as an internal diagnostic for the summary figure and
% is not reported.
%
% OUTPUT
%   F : struct with fields delta, sBc, resid, status, statusFinal,
%       sFuse, sFinal, fit, nBoth, nAgree, nConflict, nInterp, agreeRate,
%       vProf, vBin, R2, span.

    nF = numel(sP);
    both = okP & okB;

    % Estimate Delta only when there are enough co-detections.
    if sum(both) >= prm.minBothForDelta
        delta = median(sB(both) - sP(both), 'omitnan');
    else
        delta = 0;                      % cannot estimate -> do NOT invent one
    end
    sBc = sB - delta;

    resid = NaN(nF,1); resid(both) = sBc(both) - sP(both);
    agree = both & abs(resid) <= prm.agreeTol;
    confl = both & ~agree;

    status = repmat({'absent'}, nF, 1);
    sFuse  = NaN(nF,1);
    for t = 1:nF
        if agree(t)
            status{t}='agree';    sFuse(t) = 0.5*(sP(t)+sBc(t));  % independent -> mean
        elseif confl(t)
            status{t}='conflict';                                 % arbitrated below
        elseif okP(t)
            status{t}='profile';  sFuse(t) = sP(t);
        elseif okB(t)
            status{t}='binary';   sFuse(t) = sBc(t);
        end
    end

    % Global fit anchored on the CORROBORATED frames.
    ta = find(agree);
    if numel(ta) >= 3
        pf = polyfit(ta, sFuse(ta), 1);
    else
        tt = find(~isnan(sFuse));
        if numel(tt) >= 3, pf = polyfit(tt, sFuse(tt), 1); else, pf = [NaN NaN]; end
    end

    % Conflict arbitration: profile is a direct intensity measurement,
    % the binary blob is a thresholded difference. On a genuine conflict
    % take profile UNLESS it is the one that disagrees with the trajectory
    % (then binary rescues it).
    if all(isfinite(pf))
        for t = find(confl)'
            pr = polyval(pf, t);
            dP = abs(sP(t)-pr); dB = abs(sBc(t)-pr);
            if dP <= dB + prm.agreeTol, sFuse(t)=sP(t); else, sFuse(t)=sBc(t); end
        end
    end

    % HARD monotonicity: the wave never moves backward. Enforced as a
    % single global invariant (no tolerance). Any fused point behind the
    % last kept one is dropped -> it becomes 'interp' below.
    tt0 = find(~isnan(sFuse));
    if ~isempty(tt0)
        lastS = -inf;
        for t = tt0(:)'
            if sFuse(t) < lastS
                sFuse(t) = NaN;
                if strcmp(status{t},'agree'), status{t}='conflict'; end
            else
                lastS = sFuse(t);
            end
        end
    end

    % Prune ORPHAN detections: a lone fused point with no other fused
    % point within +-orphanGap frames has no trajectory to belong to.
    tt0 = find(~isnan(sFuse));
    if numel(tt0) >= 1
        for t = tt0(:)'
            near = tt0(tt0~=t & abs(tt0-t) <= prm.orphanGap);
            if isempty(near)
                sFuse(t) = NaN;
                status{t} = 'absent';
            end
        end
    end

    tt = find(~isnan(sFuse));
    sFinal = NaN(nF,1); statusFinal = status;
    if ~isempty(tt) && all(isfinite(pf))
        f0 = tt(1); f1 = tt(end);
        % Interpolate gaps BETWEEN real measurements from the measurements
        % themselves (monotone by construction), not from the global
        % straight fit which would drift on a decelerating wave. Falls
        % back to the fit only if there is a single measured point.
        for t = 1:nF
            if ~isnan(sFuse(t))
                sFinal(t) = sFuse(t);
            elseif t >= f0 && t <= f1
                if numel(tt) >= 2
                    sFinal(t) = interp1(tt, sFuse(tt), t, 'linear');  % between real pts
                else
                    sFinal(t) = polyval(pf, t);
                end
                statusFinal{t} = 'interp';
            end
        end
        % Clamp to arc + enforce monotonicity on the FINAL series too.
        lastS = -inf;
        for t = 1:nF
            if isnan(sFinal(t)), continue; end
            if sFinal(t) < lastS, sFinal(t) = lastS; end     % never backward
            lastS = sFinal(t);
        end
        F.span = [f0 f1];
    else
        F.span = [NaN NaN];
    end

    F.delta=delta; F.sBc=sBc; F.resid=resid;
    F.status=status; F.statusFinal=statusFinal;
    F.sFuse=sFuse;   F.sFinal=sFinal;  F.fit=pf;
    F.nBoth=sum(both); F.nAgree=sum(agree); F.nConflict=sum(confl);
    F.nInterp = sum(strcmp(statusFinal,'interp'));
    F.agreeRate = F.nAgree / max(F.nBoth,1);
    F.vProf  = fitSlope(find(okP), sP(okP));   % primary reported velocity
    F.vBin   = fitSlope(find(okB), sB(okB));   % independent verifier only

    % Fit quality of the PROFILE trajectory, which is the reported velocity.
    % The binary track is used only to confirm feature identity, so R2 is the
    % straight-line fit of the profile positions, not of the combined track.
    tp = find(okP);
    if numel(tp) >= 3
        yy = sP(tp); yh = polyval(polyfit(tp, yy, 1), tp);
        F.R2 = 1 - sum((yy-yh).^2)/max(sum((yy-mean(yy)).^2), eps);
    else
        F.R2 = NaN;
    end
end


function v = fitSlope(t, s)
% Linear-regression slope of s vs t; returns NaN if fewer than 3 points.
    if numel(t) < 3, v = NaN; return; end
    pf = polyfit(t(:), s(:), 1); v = pf(1);
end


% ══════════════════════════════════════════════════════════════════════


function saveAllFrames(outFolder, I, G, Pdet, Msk, cand, sigma0, ...
                       sP, snrP, okP, okB, predS, F, meta, prm)
% ONE FIGURE PER FRAME — the eye test.
% LEFT : the image, with every estimate drawn on it.
% RIGHT: the 1D profile, with the gate, the search window, every
%        candidate, and the accepted peak. When a frame goes wrong,
%        the reason is almost always visible on the right.
    fdir = fullfile(outFolder,'frames');
    if ~exist(fdir,'dir'), mkdir(fdir); end
    nF = numel(I);

    for t = 1:nF
        st = F.statusFinal{t};
        f = figure('Visible','off','Position',[100 100 1250 540]);

        % ── LEFT: image + all markers ─────────────────────────────────
        subplot(1,2,1);
        imshow(dispImage(I{t}, prm), []); hold on;   % display-only denoise
        if ~isempty(G{t})
            plot(G{t}.xm(G{t}.colValid), G{t}.ym(G{t}.colValid), 'c-', 'LineWidth', 0.9);
            if all(isfinite(F.fit)), markArc(G{t}, polyval(F.fit,t), 'k*', 9); end
            if okP(t),              markArc(G{t}, sP(t),       'b+', 16); end
            if okB(t),              markArc(G{t}, F.sBc(t),    'rx', 16); end
            if ~isnan(F.sFuse(t)),  markArc(G{t}, F.sFuse(t),  'go', 14); end
            if strcmp(st,'interp'), markArc(G{t}, F.sFinal(t), 'md', 14); end
        end
        hold off;
        title(sprintf('frame %d   [%s]', t, upper(st)), 'FontSize', 12);
        xlabel('blue+ profile   red x binary   green o FUSED   magenta d predicted   black * fit', ...
               'FontSize', 8);

        % ── RIGHT: the profile ────────────────────────────────────────
        subplot(1,2,2);
        pd = Pdet(t,:); m = Msk(t,:);
        if any(m)
            pp = pd; pp(~m) = NaN;
            plot(pp, 'r-', 'LineWidth', 1.4); hold on;

            s0 = sigma0(t);
            if isfinite(s0)
                med = median(pd(m),'omitnan');
                plot(xlim, (med + prm.snrStrong*s0)*[1 1], 'g--');
                plot(xlim, (med + prm.snrWeak  *s0)*[1 1], ':', 'Color',[0.4 0.7 0.4]);
                plot(xlim, med*[1 1], 'k-', 'LineWidth', 0.5);
            end

            % Every candidate peak that was considered.
            if ~isempty(cand{t})
                plot(cand{t}(:,1), pd(cand{t}(:,1)), 'ks', 'MarkerSize', 8);
            end

            % Where the tracker actually LOOKED.
            if isfinite(predS(t))
                yl = ylim;
                lo = predS(t) - prm.searchWin; hi = predS(t) + prm.searchWin;
                p = patch([lo hi hi lo], [yl(1) yl(1) yl(2) yl(2)], [0.88 0.9 1], ...
                          'EdgeColor','none', 'FaceAlpha',0.5);
                uistack(p,'bottom');
                plot([predS(t) predS(t)], yl, 'k--');
                ylim(yl);
            end

            if okP(t), plot(sP(t), pd(sP(t)), 'b+', 'MarkerSize',16, 'LineWidth',2); end
            if okB(t)
                xb = F.sBc(t);
                if xb >= 1 && xb <= numel(pd)
                    plot(xb, interp1(1:numel(pd), pd, xb), 'rx', 'MarkerSize',16, 'LineWidth',2);
                end
            end
            grid on; xlabel('arc s'); ylabel('detrended profile');
            if isfinite(snrP(t))
                title(sprintf('SNR = %.2f   (gates: strong %.1f, weak %.1f)', ...
                    snrP(t), prm.snrStrong, prm.snrWeak));
            else
                % Distinguish "nothing cleared the gate" from "the peak was
                % outside the search window".
                title(sprintf('no peak accepted IN WINDOW (check if a strong peak sits outside the blue band)'));
            end
            hold off;
        else
            axis off; title('profile invalid (segmentation failed)');
        end

        sgtitle(sprintf('%s | %s | %s | side %s (%s)', meta.age, meta.eye, meta.subject, ...
            meta.imgSide, upper(meta.meridian)), 'Interpreter','none', 'FontSize', 10);

        % Status IN THE FILENAME: sort the folder, go straight to the failures.
        saveas(f, fullfile(fdir, sprintf('f%03d_%s.png', t, st)));
        close(f);
    end
end


% ══════════════════════════════════════════════════════════════════════


function D = dispImage(img, prm)
% Display-only denoise. IMPORTANT: this affects ONLY the backdrop you
% look at. Detection runs on the depth-averaged profile from the RAW
% image, and every marker is computed from that real data. The image
% is easier on the eye; the verdict is identical. Set prm.dispMedian = 0
% to show raw.
    if prm.dispMedian && prm.dispMedian > 1
        D = medfilt2(img, [prm.dispMedian prm.dispMedian], 'symmetric');
    else
        D = img;
    end
end


function saveContactSheet(fpath, I, G, F, meta, prm)
% Save a grid of all frames with the final fused marker colour-coded by
% detection status. Useful for a quick eyeball pass over the whole run.
    nF = numel(I);
    nc = 8; nr = ceil(nF/nc);
    f = figure('Visible','off','Position',[50 50 1600 140*nr]);
    for t = 1:nF
        subplot(nr, nc, t);
        imshow(dispImage(I{t}, prm), []); hold on;   % display-only denoise
        if ~isempty(G{t}) && ~isnan(F.sFinal(t))
            switch F.statusFinal{t}
                case 'agree',    spec = 'go';
                case 'profile',  spec = 'bo';
                case 'binary',   spec = 'ro';
                case 'conflict', spec = 'yo';
                case 'interp',   spec = 'md';
                otherwise,       spec = '';
            end
            if ~isempty(spec), markArc(G{t}, F.sFinal(t), spec, 8); end
        end
        hold off;
        title(sprintf('%d', t), 'FontSize', 7);
    end
    sgtitle(sprintf(['%s | %s | %s | %s (%s)   ' ...
        'green=agree  blue=profile  red=binary  yellow=conflict  magenta=predicted'], ...
        meta.age, meta.eye, meta.subject, meta.imgSide, upper(meta.meridian)), ...
        'Interpreter','none','FontSize',9);
    saveas(f, fpath); close(f);
end


function markArc(G, s, spec, sz)
% Map an arc position back onto the image so it can be drawn on the frame.
% Uses the geometry's (s, xm, ym) lookup to convert arc -> (x, y).
    if isnan(s), return; end
    x = interp1(G.s, G.xm, s, 'linear', NaN);
    y = interp1(G.s, G.ym, s, 'linear', NaN);
    if isnan(x), return; end
    plot(x, y, spec, 'MarkerSize', sz, 'LineWidth', 2);
end


% ══════════════════════════════════════════════════════════════════════


function saveFusionDiag(fpath, sP, okP, sB, okB, F, meta, prm, phiMed)
% Save a 2x2 diagnostic figure summarising fusion quality:
%   1. Two measurements (profile, binary raw, binary bias-corrected).
%   2. Per-frame residuals with agreeTol bands.
%   3. Final fused trajectory with status colour coding.
%   4. Verdict text panel with speeds, agreement counts, and grade.
    f = figure('Visible','off','Position',[100 100 1500 900]);

    subplot(2,2,1); hold on;
    plot(find(okB), sB(okB),    'rs--','MarkerSize',5,'DisplayName','binary (raw)');
    plot(find(okB), F.sBc(okB), 'r^-', 'MarkerSize',5,'DisplayName','binary (bias-corrected)');
    plot(find(okP), sP(okP),    'bo-', 'MarkerSize',5,'LineWidth',1.4,'DisplayName','profile');
    grid on; xlabel('frame'); ylabel('arc s'); legend('Location','best');
    title(sprintf('Two independent measurements  (\\Delta = %+.1f px)', F.delta));
    hold off;

    subplot(2,2,2); hold on;
    b = find(okP & okB);
    if ~isempty(b), stem(b, F.resid(b), 'k','filled','MarkerSize',4); end
    plot(xlim,  prm.agreeTol*[1 1],'g--'); plot(xlim, -prm.agreeTol*[1 1],'g--');
    plot(xlim,[0 0],'k-'); grid on;
    xlabel('frame'); ylabel('binary_{corr} - profile  (arc px)');
    title(sprintf('Agreement: %d/%d within \\pm%.0f px  (%.0f%%)', ...
        F.nAgree, F.nBoth, prm.agreeTol, 100*F.agreeRate));
    hold off;

    subplot(2,2,3); hold on;
    for t = 1:numel(F.sFinal)
        if isnan(F.sFinal(t)), continue; end
        switch F.statusFinal{t}
            case 'agree',    plot(t,F.sFinal(t),'o','MarkerSize',6,'MarkerFaceColor',[0 .6 0],'MarkerEdgeColor','none');
            case 'profile',  plot(t,F.sFinal(t),'o','MarkerSize',6,'MarkerFaceColor',[0 0 1],'MarkerEdgeColor','none');
            case 'binary',   plot(t,F.sFinal(t),'o','MarkerSize',6,'MarkerFaceColor',[1 0 0],'MarkerEdgeColor','none');
            case 'conflict', plot(t,F.sFinal(t),'o','MarkerSize',7,'MarkerFaceColor',[.9 .7 0],'MarkerEdgeColor','k');
            case 'interp',   plot(t,F.sFinal(t),'d','MarkerSize',6,'MarkerEdgeColor',[1 0 1],'LineWidth',1.2);
        end
    end
    if all(isfinite(F.fit))
        tt = 1:numel(F.sFinal); plot(tt, polyval(F.fit,tt), 'k-', 'LineWidth',1.1);
    end
    grid on; xlabel('frame'); ylabel('final arc s');
    title(sprintf('PROFILE velocity = %.2f px/frame, R^2 = %.3f   (markers = combined diagnostic track)', ...
        F.vProf, F.R2));
    hold off;

    subplot(2,2,4); axis off;
    R = packResult(meta, numel(sP), okP, okB, F, phiMed, '', prm);
    txt = {sprintf('%s | %s | %s', meta.age, meta.eye, meta.subject), ...
           sprintf('image side %s  ->  %s', meta.imgSide, upper(meta.meridian)), ...
           '----------------------------------------', ...
           sprintf('profile   : %d frames', sum(okP)), ...
           sprintf('binary    : %d frames', sum(okB)), ...
           sprintf('both      : %d', F.nBoth), ...
           sprintf('  agree   : %d  (%.0f%%)', F.nAgree, 100*F.agreeRate), ...
           sprintf('  conflict: %d', F.nConflict), ...
           sprintf('interp    : %d  (PREDICTED, not measured)', F.nInterp), ...
           sprintf('span      : frames %g - %g', F.span(1), F.span(2)), ...
           '----------------------------------------', ...
           sprintf('leading-edge bias  Delta = %+.1f px', F.delta), ...
           '  binary sits AHEAD of the profile centre.', ...
           '  WARNING: a 1-frame offset looks IDENTICAL.', ...
           '----------------------------------------', ...
           sprintf('v profile : %.2f px/frame  (R2 = %.3f)  [reported]', F.vProf, F.R2), ...
           sprintf('v binary  : %.2f px/frame  (verifier only)', F.vBin), ...
           sprintf('phi       : %.1f deg from the normal', phiMed), ...
           '----------------------------------------', ...
           sprintf('GRADE: %s', R.grade)};
    text(0.02,0.98, txt, 'VerticalAlignment','top','FontSize',10,'FontName','Courier New');
    title('Verdict');

    saveas(f, fpath); close(f);
end


% ══════════════════════════════════════════════════════════════════════


function R = packResult(meta, nF, okP, okB, F, phiMed, offsetWarn, prm)
% Pack the per-(subject, side) summary into one struct that the CSV writer
% and the master summary plots consume.
%
% Grading thresholds are HEURISTIC. Calibrate them from the agreement
% histogram in master_summary.png before reading the grades as truth.
%   A : >=8 agreeing frames, >=80% agreement, profile/binary speeds agree.
%   B : >=5 agreeing frames, >=50% agreement.
%   C : the profile track has >=8 detections and a finite velocity.
%   D : a finite profile velocity exists but with weaker evidence.
%   F : no usable profile velocity.
    R = meta;
    R.nFrames=nF; R.nProfile=sum(okP); R.nBinary=sum(okB);
    R.nBoth=F.nBoth; R.nAgree=F.nAgree; R.nConflict=F.nConflict; R.nInterp=F.nInterp;
    R.agreeRate=F.agreeRate; R.delta=F.delta;
    R.vProf=F.vProf; R.vBin=F.vBin; R.R2=F.R2;
    R.phi=phiMed; R.offsetWarn=offsetWarn;
    R.spanStart=F.span(1); R.spanEnd=F.span(2);

    if isfinite(prm.mmPerPx) && isfinite(prm.msPerFrame)
        R.vProf_m_s = F.vProf * prm.mmPerPx / prm.msPerFrame;
    else
        R.vProf_m_s = NaN;
    end

    % Grade on the PROFILE track (the reported velocity). The binary track
    % enters only through the agreement statistics that confirm the two
    % methods followed the same feature.
    vAgree = isfinite(F.vProf) && isfinite(F.vBin) && ...
             abs(F.vProf-F.vBin)/max(abs(F.vProf),eps) < 0.20;
    if     F.nAgree>=8 && F.agreeRate>=0.8 && vAgree,   R.grade='A';
    elseif F.nAgree>=5 && F.agreeRate>=0.5,             R.grade='B';
    elseif R.nProfile>=8 && isfinite(F.vProf),          R.grade='C';
    elseif isfinite(F.vProf),                           R.grade='D';
    else,                                               R.grade='F';
    end
    % Reject absurd results: bound the reported profile velocity to the
    % admissible speed band and require a minimum yield. The RANSAC seed is
    % constrained to [vMin,vMax] but the final least-squares slope is not,
    % which is how a 3-frame track produced 13.0 px/frame. Rejected rows get
    % NaN velocities and grade 'R', so the statistics drop them.
    reason = '';
    if ~isfinite(R.vProf)
        reason = 'no_velocity';
    elseif R.vProf > prm.vMax
        reason = sprintf('v_too_fast_%.1f', R.vProf);
    elseif R.nProfile < prm.minValidFrames
        reason = sprintf('low_yield_%d', R.nProfile);
    end
    if ~isempty(reason)
        R.vProf = NaN; R.vBin = NaN; R.vProf_m_s = NaN; R.R2 = NaN;
        R.grade = 'R';
    end
    R.rejectReason = reason;
end


% ══════════════════════════════════════════════════════════════════════


function writeMasterCSV(fpath, rows)
% Write every packed result row to a single CSV at fpath.
    fid = fopen(fpath,'w');
    if fid < 0, warning('cannot write %s', fpath); return; end
    fprintf(fid, ['age,eye,subject,img_side,meridian,n_frames,n_profile,n_binary,n_both,' ...
                  'n_agree,n_conflict,n_interp,agree_rate,delta_px,span_start,span_end,' ...
                  'v_profile,v_profile_m_s,v_binary,R2,phi_deg,grade,offset_warning\n']);
    for k = 1:numel(rows)
        r = rows{k};
        fprintf(fid, '%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.3f,%.2f,%g,%g,%.3f,%.4f,%.3f,%.3f,%.1f,%s,%s\n', ...
            r.age,r.eye,r.subject,r.imgSide,r.meridian,r.nFrames,r.nProfile,r.nBinary, ...
            r.nBoth,r.nAgree,r.nConflict,r.nInterp,r.agreeRate,r.delta,r.spanStart,r.spanEnd, ...
            r.vProf,r.vProf_m_s,r.vBin,r.R2,r.phi,r.grade,r.offsetWarn);
    end
    fclose(fid);
end


% ══════════════════════════════════════════════════════════════════════


function saveMasterSummary(fpath, rows)
% Save a 2x2 master summary figure aggregating the grades, agreement
% rates, and per-group speeds. Calibrate the grading thresholds from
% the agreement histogram before trusting the grades.
    if isempty(rows), return; end
    f = figure('Visible','off','Position',[100 100 1500 900]);
    grades = cellfun(@(r) r.grade, rows);
    ages   = cellfun(@(r) string(r.age), rows);
    merid  = cellfun(@(r) string(r.meridian), rows);
    vF     = cellfun(@(r) r.vProf, rows);
    agr    = cellfun(@(r) r.agreeRate, rows);

    subplot(2,2,1);
    histogram(categorical(cellstr(grades'))); grid on;
    ylabel('sequences'); title('Grades (A = both methods corroborate)');

    subplot(2,2,2);
    histogram(agr, 0:0.1:1); grid on;
    xlabel('agreement rate'); ylabel('sequences');
    title('Binary-vs-profile agreement — SET THE GRADE THRESHOLDS FROM THIS');

    ok = isfinite(vF(:)) & (grades(:)=='A' | grades(:)=='B');
    subplot(2,2,3);
    if any(ok), groupScatter(ages(ok), vF(ok)); ylabel('v (arc px/frame)');
        title('Speed by AGE (A/B only)'); grid on;
    else, axis off; title('no A/B sequences'); end

    subplot(2,2,4);
    if any(ok), groupScatter(merid(ok), vF(ok)); ylabel('v (arc px/frame)');
        title('Speed by MERIDIAN (A/B only)'); grid on;
    else, axis off; end

    saveas(f, fpath); close(f);
end


function groupScatter(groups, v)
% Scatter v against categorical groups, with a red median bar per group.
    g = unique(groups); hold on;
    for i = 1:numel(g)
        m = groups == g(i);
        scatter(i + 0.12*randn(sum(m),1), v(m), 22, [.3 .3 .8], 'filled', 'MarkerFaceAlpha',0.6);
        plot([i-0.25 i+0.25], median(v(m),'omitnan')*[1 1], 'r-', 'LineWidth',2);
    end
    xlim([0.5 numel(g)+0.5]);
    set(gca,'XTick',1:numel(g),'XTickLabel',cellstr(g)); hold off;
end


% ══════════════════════════════════════════════════════════════════════


function lab = nasalTemporal(eyeName, imgSide, rightEyeImageRightIs)
% Translate the (eye, image side) pair into an anatomical meridian label
% (nasal or temporal). One convention fixes all four combinations
% (the eyes are mirror images).
%
% INPUTS
%   eyeName              : string containing 'RIGHT' or 'LEFT'.
%   imgSide              : 'R' or 'L' (image side stored).
%   rightEyeImageRightIs : 'temporal' or 'nasal' (the convention set in prm).
%
% OUTPUT
%   lab                  : 'NASAL' or 'TEMPORAL'.
    if strcmpi(rightEyeImageRightIs,'nasal'), other='temporal'; else, other='nasal'; end
    isRight = ~isempty(strfind(upper(eyeName),'RIGHT')); %#ok<STREMP>
    if isRight
        if imgSide=='R', lab=rightEyeImageRightIs; else, lab=other; end
    else
        if imgSide=='R', lab=other; else, lab=rightEyeImageRightIs; end
    end
end


function nm = oldSideName(eyeName, imgSide)
% Build the legacy short tag combining eye and image side (e.g. 'RoR',
% 'LoR', 'RoL', 'LoL') used for the tracker output folder.
    isRight = ~isempty(strfind(upper(eyeName),'RIGHT')); %#ok<STREMP>
    if isRight
        if imgSide=='R', nm='RoR'; else, nm='LoR'; end
    else
        if imgSide=='R', nm='RoL'; else, nm='LoL'; end
    end
end


function d = listDirs(p)
% Return the names of subdirectories of p (excluding '.' and '..').
    e = dir(p);
    e = e([e.isdir] & ~ismember({e.name},{'.','..'}));
    d = {e.name};
end


% ══════════════════════════════════════════════════════════════════════
% End of file
% ══════════════════════════════════════════════════════════════════════



% ══════════════════════════════════════════════════════════════════════
% SYNTHETIC GROUND-TRUTH VERIFICATION  (merged from the former verification.m)
% The 'verify' mode builds synthetic sequences of known velocity and runs
% them through the same detection functions used on the real data.
% ══════════════════════════════════════════════════════════════════════

function verifyMode(outRoot, prm)
    vdir = fullfile(outRoot, 'verification');
    if ~exist(vdir,'dir'), mkdir(vdir); end
    k = prm.mmPerPx / prm.msPerFrame;
    fid = fopen(fullfile(vdir,'verification_log.txt'), 'w');
    logf(fid, 'SYNTHETIC GROUND-TRUTH VERIFICATION\n');
    logf(fid, 'calibration k = %.6f  (m/s per arc px/frame)\n', k);
    logf(fid, 'each synthetic feature moves at a known arc velocity along the true midline\n\n');

    kcurvNom = 0.004;   noiseNom = 8;   ampNom = 60;

    % 0) SYNTHETIC DATA EXAMPLE
    [I_synth, truth_synth] = synthSequence(2.5, kcurvNom, noiseNom, ampNom, prm, 999);
    saveSyntheticFigures(fullfile(vdir,'fig_synthetic_data.png'), I_synth, truth_synth);

    % 1) VELOCITY SWEEP
    vList = [1.5 2.5 3.5 4.5 5.5 6.5];   nTrial = 5; 
    logf(fid, '== velocity sweep (kcurv=%.3f, noise=%.1f, amp=%.0f, %d trials/level) ==\n', kcurvNom, noiseNom, ampNom, nTrial);
    Vtrue=[]; Vrec=[]; Vprof=[]; Vbin=[]; Varc=[];
    for iv = 1:numel(vList)
        for it = 1:nTrial
            [I, truth] = synthSequence(vList(iv), kcurvNom, noiseNom, ampNom, prm, 100*iv+it);
            o = runCoreQuiet(I, prm);
            if isempty(o), logf(fid,'  vTrue=%.2f trial %d: RUN FAILED\n', vList(iv), it); continue; end
            r = o.arcLen / truth.sMax;
            Vtrue(end+1)=vList(iv); Vrec(end+1)=o.vRec; Vprof(end+1)=o.vProf; %#ok<AGROW>
            Vbin(end+1)=o.vBin;     Varc(end+1)=r; %#ok<AGROW>
            logf(fid, '  vTrue=%.2f trial %d: vProf=%.3f (bin %.3f) R2=%.2f nP=%d nB=%d arcRatio=%.3f SNR=%.1f\n', ...
                vList(iv), it, o.vRec, o.vBin, o.R2, o.nProf, o.nBin, r, o.medSNR);
        end
    end
    [sl, ic, rmse] = linfit(Vtrue, Vrec);
    logf(fid, '\n  profile recovery:  recovered = %.3f * true %+.3f   (RMSE %.3f px/frame = %.4f m/s)\n', sl, ic, rmse, rmse*k);
    logf(fid, '  mean arc-length ratio (pipeline/true) = %.3f  (slope<1 is largely explained by this)\n\n', mean(Varc,'omitnan'));
    verifyScatter(fullfile(vdir,'fig_recovery_velocity.png'), Vtrue, Vrec, sl, ic, ...
        'prescribed velocity (arc px/frame)', 'recovered profile velocity (arc px/frame)', 'Velocity recovery (profile)');
    writeVerifyCSV(fullfile(vdir,'verification_velocity.csv'), ...
        {'v_true_pxf','v_rec_pxf','v_prof_pxf','v_bin_pxf','arc_ratio'}, ...
        [col(Vtrue) col(Vrec) col(Vprof) col(Vbin) col(Varc)]);

    % 2) NOISE SWEEP
    nTrialN = 5; 
    logf(fid, '== noise sweep (vTrue=2.50, kcurv=%.3f, amp=%.0f, %d trials/level) ==\n', kcurvNom, ampNom, nTrialN);
    noiseList = [5 7 9 11 13 15 18 22];   % sigma=3 dropped: low-noise degeneracy below real acquisition noise
    Nx=[]; NmeanErr=[]; NsdErr=[]; NmeanYld=[]; Nraw=[];
    for in = 1:numel(noiseList)
        errs=[]; ylds=[]; snrs=[];
        for it = 1:nTrialN
            [I, ~] = synthSequence(2.5, kcurvNom, noiseList(in), ampNom, prm, 500+10*in+it);
            o = runCoreQuiet(I, prm);
            if isempty(o), continue; end
            errs(end+1)=o.vRec-2.5; ylds(end+1)=o.nProf; snrs(end+1)=o.medSNR; %#ok<AGROW>
            Nraw = [Nraw; noiseList(in) o.vRec o.vRec-2.5 o.nProf o.medSNR]; %#ok<AGROW>
        end
        if isempty(errs), logf(fid,'  noise=%2.0f: all runs failed\n', noiseList(in)); continue; end
        Nx(end+1)=noiseList(in); NmeanErr(end+1)=mean(errs); NsdErr(end+1)=std(errs); NmeanYld(end+1)=mean(ylds); %#ok<AGROW>
        logf(fid, '  noise=%2.0f: err = %+.3f +/- %.3f px/frame   (yield nP=%.0f, medSNR=%.0f, n=%d)\n', ...
            noiseList(in), mean(errs), std(errs), mean(ylds), median(snrs), numel(errs));
    end
    verifyNoisePanel(fullfile(vdir,'fig_recovery_snr.png'), Nx, NmeanErr, NsdErr, NmeanYld, 26, ...
        sprintf('Robustness to noise (v_{true} = 2.5, %d trials/level)', nTrialN));
    if isempty(Nraw), Nraw = nan(1,5); end
    writeVerifyCSV(fullfile(vdir,'verification_snr.csv'), ...
        {'noise_sigma','v_rec_pxf','err_pxf','n_profile','med_snr'}, Nraw);

    % 3) CURVATURE SWEEP
    curvList = [0.001 0.003 0.005 0.007 0.009];
    nTrialC = 3; 
    logf(fid, '\n== curvature sweep (vTrue=2.50, noise=%.1f, amp=%.0f, %d trials/level) ==\n', noiseNom, ampNom, nTrialC);
    Carc=[]; Crec=[]; Cerr=[];
    for ic2 = 1:numel(curvList)
        for it = 1:nTrialC
            [I, truth] = synthSequence(2.5, curvList(ic2), noiseNom, ampNom, prm, 8000+100*ic2+it);
            o = runCoreQuiet(I, prm);
            if isempty(o), logf(fid,'  kcurv=%.3f trial %d: RUN FAILED\n', curvList(ic2), it); continue; end
            Carc(end+1)=truth.sMax; Crec(end+1)=o.vRec; Cerr(end+1)=o.vRec-2.5; %#ok<AGROW>
            logf(fid, '  kcurv=%.3f trial %d, arcLen=%5.1f: vProf=%.3f  err=%+.3f  arcRatio=%.3f\n', ...
                curvList(ic2), it, truth.sMax, o.vRec, o.vRec-2.5, o.arcLen/truth.sMax);
        end
    end
    verifyLine(fullfile(vdir,'fig_recovery_curvature.png'), Carc, Crec, 2.5, ...
        'true arc length (px)', 'recovered profile velocity (arc px/frame)', 'Robustness to curvature (vTrue = 2.5)');
    writeVerifyCSV(fullfile(vdir,'verification_curvature.csv'), {'arc_len','v_rec_pxf','err_pxf'}, ...
        [col(Carc) col(Crec) col(Cerr)]);

    if fid > 0, fclose(fid); end
    fprintf('\n  Verification outputs written to: %s\n\n', vdir);
end


function [I, truth] = synthSequence(vTrue, kcurv, noiseSigma, featAmp, prm, seed, accel)
    if nargin < 6 || isempty(seed), seed = 0; end
    if nargin < 7 || isempty(accel), accel = 0; end
    rng(seed);
    W  = prm.cropRect(3) + 1;
    H  = prm.cropRect(4) + 1;
    th = 24;
    xc = (W+1)/2;   x = 1:W;
    aTop = 42;
    a = aTop + kcurv .* (x - xc).^2;
    if max(a) + th > H - 12
        kcurv = max(((H-12) - th - aTop) / max((x-xc).^2), 0);
        a = aTop + kcurv .* (x - xc).^2;
    end
    p = a + th;   m = a + th/2;
    dm = gradient(m);   s = cumsum(sqrt(1 + dm.^2));   s = s - s(1);   sMax = s(end);
    s0 = 24;
    nMax = 60;   vmin = 1.0;   taus = 0:nMax;
    sfs = s0 + vTrue*taus + 0.5*accel*taus.^2;
    vfs = vTrue + accel*taus;
    active = (sfs <= sMax - 6) & (vfs >= vmin) & (sfs >= s0 - 1);
    la = find(active, 1, 'last');   if isempty(la), la = 22; end
    nF = min(max(la + 2, 22), nMax);
    [CC, RR] = meshgrid(1:W, 1:H);
    aM = repmat(a, H, 1);   pM = repmat(p, H, 1);   inBand = RR >= aM & RR <= pM;
    band = 120;
    rc = min(round(max(p) + 12), H - 4);   cc = round(xc);   rad = 8;
    refl = ((CC - cc).^2 + (RR - rc).^2) <= rad^2;
    phi0 = deg2rad(20);   tanphi = tan(phi0);   sigS = 3.0;   sigD = th/2.2;
    I = cell(nF,1);   sfeat = nan(nF,1);
    for t = 1:nF
        img = 22 * ones(H, W);
        img(inBand) = band;
        img(refl)   = 200;
        tau = t - 1;
        sf = s0 + vTrue*tau + 0.5*accel*tau^2;   sfeat(t) = sf;
        if sf <= sMax - 2 && (vTrue + accel*tau) >= 0.3
            xf = interp1(s, x, sf, 'linear', NaN);
            if isfinite(xf)
                mf = interp1(x, m, xf, 'linear', m(min(max(round(xf),1),W)));
                cCentre = xf - (RR - mf) .* tanphi;
                feat = featAmp .* exp(-((CC - cCentre).^2)/(2*sigS^2) - ((RR - mf).^2)/(2*sigD^2));
                img = img + feat .* inBand;
            end
        end
        img = img + noiseSigma * randn(H, W);
        I{t} = min(max(img, 0), 255);
    end
    truth = struct('vTrue',vTrue, 'v0',vTrue, 'accel',accel, 'sMax',sMax, 'nF',nF, 's0',s0, ...
                   'kcurv',kcurv, 'phi0deg',20, 'mid', m, 'ant', a);
end


function o = runCoreQuiet(I, prm)
    o = [];
    try
        evalc('o = runCore(I, prm);');   %#ok<NASGU>
    catch
        o = [];
    end
end


function saveSyntheticFigures(fpath, I, truth)
    nF = numel(I);
    t1 = max(5, round(nF * 0.2));
    t2 = max(10, round(nF * 0.5));
    t3 = min(nF, round(nF * 0.8));
    f = figure('Visible','off','Position',[100 100 1200 400]);
    subplot(1,3,1);
    imshow(uint8(I{t1})); title(sprintf('Synthetic Frame %d (Early)', t1));
    subplot(1,3,2);
    imshow(uint8(I{t2})); hold on;
    if isfield(truth, 'mid')
        plot(1:numel(truth.mid), truth.mid, 'g--', 'LineWidth', 1.5);
        plot(1:numel(truth.ant), truth.ant, 'r--', 'LineWidth', 1.0);
        legend({'True Midline', 'True Anterior'}, 'Location', 'south', 'FontSize', 7);
    end
    title(sprintf('Synthetic Frame %d (Mid)', t2));
    subplot(1,3,3);
    imshow(uint8(I{t3})); title(sprintf('Synthetic Frame %d (Late)', t3));
    sgtitle('Synthetic Corvis-ST Data (Ground Truth Wave Propagation)', 'FontSize', 14);
    saveas(f, fpath); close(f);
end


function o = runCore(I, prm)
    s2 = prm.cropRect(3) + 1;   nF = numel(I);
    [A, P] = segmentAllFrames(I, prm);
    [A, P] = temporalSmoothSurfaces(A, P, prm);
    G = cell(nF,1);
    for t = 1:nF
        if all(~isfinite(A(t,:))), G{t} = []; else, G{t} = buildGeometry(A(t,:), P(t,:), prm); end
    end
    thkAll = P - A;   rect_H = round(median(thkAll(:), 'omitnan'));
    if ~isfinite(rect_H) || rect_H < 5, rect_H = 20; end

    warped = cell(nF,1);
    for t = 1:nF
        if isempty(G{t}), warped{t} = zeros(rect_H, s2);
        else,             warped{t} = warpFrame(I{t}, G{t}, rect_H, prm); end
    end
    bg       = buildBGStack(warped);
    strict   = strictBinaryFromWarped(warped, prm);
    binTrack = trackComponentInMem(strict, bg, '', false);

    [sigma0, phiMed] = phiScan(I, G, prm);
    [Pdet, Msk, cand, totalArc] = buildProfiles(I, G, sigma0, phiMed, nF, prm);
    [line0, inl] = ransacTrajectory(cand, prm);
    if isempty(line0)
        sP = NaN(nF,1); snrP = NaN(nF,1); okP = false(nF,1); predS = NaN(nF,1);
    else
        [sP, snrP, okP, predS] = refineTrajectory(Pdet, Msk, sigma0, line0, inl, prm); %#ok<ASGLU>
    end

    sB = NaN(nF,1); okB = false(nF,1); rectW = prm.cropRect(3) + 1;
    if ~isempty(binTrack) && isfield(binTrack,'centroid')
        xb = binTrack.centroid(:,1);
        for t = 1:nF
            if t <= numel(xb) && ~isnan(xb(t)) && ~isempty(G{t}) && isfinite(totalArc(t))
                sB(t) = (xb(t)-1)/(rectW-1) * totalArc(t); okB(t) = true;
            end
        end
    end
    
    F = fuseTracks(sP, okP, sB, okB, prm);

    t_prof = find(okP);
    if numel(t_prof) >= 3
        s_prof = sP(okP);
        p1 = polyfit(t_prof, s_prof, 1);
        yh = polyval(p1, t_prof);
        R2Prof = 1 - sum((s_prof - yh).^2) / max(sum((s_prof - mean(s_prof)).^2), eps);
    else
        R2Prof = NaN;
    end

    o = struct('vRec', F.vProf, 'vProf', F.vProf, 'vBin', F.vBin, 'R2', R2Prof, ...
               'nProf', sum(okP), 'nBin', sum(okB), ...
               'arcLen', median(totalArc(isfinite(totalArc))), ...
               'medSNR', median(snrP(okP),'omitnan'), 'phiMed', phiMed);
end


function [slope, intercept, rmse] = linfit(x, y)
    x = x(:); y = y(:); ok = isfinite(x) & isfinite(y); x = x(ok); y = y(ok);
    if numel(x) < 2, slope = NaN; intercept = NaN; rmse = NaN; return; end
    pf = polyfit(x, y, 1); slope = pf(1); intercept = pf(2);
    rmse = sqrt(mean((y - polyval(pf, x)).^2));
end


function c = col(v)
    c = v(:);
end


function verifyScatter(fpath, xt, yr, slope, intercept, xl, yl, ttl)
    xt = xt(:); yr = yr(:);
    ok = isfinite(xt) & isfinite(yr); xt = xt(ok); yr = yr(ok);
    if isempty(xt), return; end
    f = figure('Visible','off','Position',[100 100 520 500]);
    scatter(xt, yr, 32, [0.20 0.35 0.75], 'filled', 'MarkerFaceAlpha', 0.6); hold on;
    lo = min([xt; yr]); hi = max([xt; yr]);
    plot([lo hi], [lo hi], 'k--', 'LineWidth', 1.0);
    xx = linspace(min(xt), max(xt), 50);
    plot(xx, slope*xx + intercept, 'r-', 'LineWidth', 1.4);
    hold off; grid on; axis equal;
    xlabel(xl); ylabel(yl);
    title(sprintf('%s: slope %.3f, intercept %+.3f', ttl, slope, intercept));
    legend({'sequences','ideal (y = x)','least-squares fit'}, 'Location','northwest', 'FontSize', 8);
    saveas(f, fpath); close(f);
end


function verifyLine(fpath, xv, yr, vtrue, xl, yl, ttl)
    xv = xv(:); yr = yr(:);
    ok = isfinite(xv) & isfinite(yr); xv = xv(ok); yr = yr(ok);
    if isempty(xv), return; end
    [xs, ord] = sort(xv); ys = yr(ord);
    f = figure('Visible','off','Position',[100 100 560 440]);
    plot(xs, ys, 'o-', 'Color',[0.20 0.35 0.75], 'MarkerFaceColor',[0.20 0.35 0.75], 'LineWidth',1.3); hold on;
    plot([min(xs) max(xs)], vtrue*[1 1], 'k--', 'LineWidth', 1.0);
    ylim([vtrue - 0.5, vtrue + 0.5]); 
    hold off; grid on; xlabel(xl); ylabel(yl); title(ttl);
    legend({'recovered','prescribed'}, 'Location','best', 'FontSize', 8);
    saveas(f, fpath); close(f);
end


function writeVerifyCSV(fpath, headers, M)
    fid = fopen(fpath, 'w');   if fid < 0, return; end
    fprintf(fid, '%s', headers{1});
    for j = 2:numel(headers), fprintf(fid, ',%s', headers{j}); end
    fprintf(fid, '\n');
    for i = 1:size(M,1)
        fprintf(fid, '%.6g', M(i,1));
        for j = 2:size(M,2), fprintf(fid, ',%.6g', M(i,j)); end
        fprintf(fid, '\n');
    end
    fclose(fid);
end


function verifyErrBand(fpath, x, y, e, xl, yl, ttl)
    x = x(:); y = y(:); e = e(:);
    ok = isfinite(x) & isfinite(y); x = x(ok); y = y(ok); e = e(ok);
    if isempty(x), return; end
    [xs, ord] = sort(x); ys = y(ord); es = e(ord);
    f = figure('Visible','off','Position',[100 100 560 440]);
    errorbar(xs, ys, es, 'o-', 'Color',[0.20 0.35 0.75], ...
             'MarkerFaceColor',[0.20 0.35 0.75], 'LineWidth',1.3); hold on;
    plot(xlim, [0 0], 'k--', 'LineWidth', 1.0);
    hold off; grid on; xlabel(xl); ylabel(yl); title(ttl);
    legend({'mean \pm SD error','zero error'}, 'Location','best', 'FontSize', 8);
    saveas(f, fpath); close(f);
end


function logf(fid, varargin)
    fprintf(varargin{:});
    if fid > 0, fprintf(fid, varargin{:}); end
end


function verifyNoisePanel(fpath, x, err, sd, yld, realYield, ttl)
    % Two-panel noise-robustness figure.
    %   top    : mean +/- SD velocity error vs injected noise
    %   bottom : profile detection yield vs injected noise, with the
    %            real-data median yield drawn for reference.
    x = x(:); err = err(:); sd = sd(:); yld = yld(:);
    ok = isfinite(x) & isfinite(err); x = x(ok); err = err(ok); sd = sd(ok); yld = yld(ok);
    if isempty(x), return; end
    [x, ord] = sort(x); err = err(ord); sd = sd(ord); yld = yld(ord);

    col   = [0.20 0.35 0.75];
    bandc = [0.72 0.80 0.94];
    xpad  = [min(x)-1, max(x)+1];

    f = figure('Visible','off','Position',[100 100 640 640],'Color','w');

    % ---- top panel: velocity error ----
    ax1 = subplot(2,1,1); hold(ax1,'on'); box(ax1,'on');
    hb = fill([x; flipud(x)], [err-sd; flipud(err+sd)], bandc, ...
              'EdgeColor','none', 'FaceAlpha',0.6, 'Parent',ax1);
    plot(ax1, xpad, [0 0], 'k--', 'LineWidth',1.0);
    hm = plot(ax1, x, err, '-o', 'Color',col, 'MarkerFaceColor',col, ...
              'LineWidth',1.8, 'MarkerSize',6);
    hold(ax1,'off'); grid(ax1,'on'); xlim(ax1, xpad);
    ylabel(ax1, 'velocity error (arc px/frame)', 'FontSize',12);
    title(ax1, ttl, 'FontSize',13, 'FontWeight','bold');
    set(ax1, 'FontSize',11, 'XTickLabel',[]);
    legend([hm hb], {'mean error','\pm SD'}, 'Location','northwest', 'FontSize',9);

    % ---- bottom panel: detection yield ----
    ax2 = subplot(2,1,2); hold(ax2,'on'); box(ax2,'on');
    hy = plot(ax2, x, yld, '-s', 'Color',[0.30 0.30 0.30], ...
              'MarkerFaceColor',[0.30 0.30 0.30], 'LineWidth',1.8, 'MarkerSize',6);
    if nargin >= 6 && ~isempty(realYield) && isfinite(realYield)
        hr = plot(ax2, xpad, [realYield realYield], ':', ...
                  'Color',[0.70 0.20 0.20], 'LineWidth',1.4);
        legend([hy hr], {'profile yield n_P','real-data median yield'}, ...
               'Location','northeast', 'FontSize',9);
    end
    hold(ax2,'off'); grid(ax2,'on');
    xlim(ax2, xpad); ylim(ax2, [0, max(yld)*1.15]);
    xlabel(ax2, 'injected noise \sigma (gray levels)', 'FontSize',12);
    ylabel(ax2, 'profile yield n_P (frames)', 'FontSize',12);
    set(ax2, 'FontSize',11);

    print(f, fpath, '-dpng', '-r200'); close(f);
end
