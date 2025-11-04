/**
 * SAM Browser ONNX Demo - Main Application
 * Handles ONNX Runtime Web inference, canvas interactions, and UI updates.
 */

// Global state
const state = {
    currentModel: null,
    currentImage: null,
    imageId: null,
    imageEmbedding: null,
    onnxSession: null,
    clicks: [],
    canvas: null,
    ctx: null,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    isDragging: false,
    lastX: 0,
    lastY: 0,
    dilation: 0,
    currentMask: null,
    debounceTimer: null,
};

// UI Elements
const elements = {
    modelSelect: document.getElementById('modelSelect'),
    imageUpload: document.getElementById('imageUpload'),
    canvas: document.getElementById('canvas'),
    undoBtn: document.getElementById('undoBtn'),
    clearBtn: document.getElementById('clearBtn'),
    exportBtn: document.getElementById('exportBtn'),
    dilationSlider: document.getElementById('dilationSlider'),
    dilationValue: document.getElementById('dilationValue'),
    modelStatus: document.getElementById('modelStatus'),
    uploadStatus: document.getElementById('uploadStatus'),
    imageInfo: document.getElementById('imageInfo'),
    clickCount: document.getElementById('clickCount'),
    maskArea: document.getElementById('maskArea'),
    runtimeInfo: document.getElementById('runtimeInfo'),
    toast: document.getElementById('toast'),
    loadingOverlay: document.getElementById('loadingOverlay'),
};

// Initialize
async function init() {
    state.canvas = elements.canvas;
    state.ctx = state.canvas.getContext('2d');

    // Setup event listeners
    elements.modelSelect.addEventListener('change', handleModelChange);
    elements.imageUpload.addEventListener('change', handleImageUpload);
    elements.undoBtn.addEventListener('click', handleUndo);
    elements.clearBtn.addEventListener('click', handleClear);
    elements.exportBtn.addEventListener('click', handleExport);
    elements.dilationSlider.addEventListener('input', handleDilationChange);

    // Canvas interactions
    elements.canvas.addEventListener('mousedown', handleMouseDown);
    elements.canvas.addEventListener('mousemove', handleMouseMove);
    elements.canvas.addEventListener('mouseup', handleMouseUp);
    elements.canvas.addEventListener('wheel', handleWheel, { passive: false });
    elements.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    // Check ONNX Runtime
    try {
        const ortVersion = ort.env.versions.common;
        elements.runtimeInfo.textContent = `v${ortVersion}`;

        // Try to detect WebGPU support
        if (navigator.gpu) {
            elements.runtimeInfo.textContent += ' (WebGPU available)';
        } else {
            elements.runtimeInfo.textContent += ' (WASM fallback)';
        }
    } catch (e) {
        showToast('Failed to initialize ONNX Runtime', 'error');
    }

    // Load available models
    await loadAvailableModels();
}

// Load available models from server
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();

        // Update model select options
        elements.modelSelect.innerHTML = '<option value="">Select model...</option>';
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model.replace('vit_', 'ViT-').replace('_b', 'B').replace('_l', 'L').replace('_h', 'H');
            elements.modelSelect.appendChild(option);
        });

        if (data.models.length === 0) {
            showToast('No ONNX models found. Please export models first.', 'error');
        }
    } catch (e) {
        showToast('Failed to load available models', 'error');
    }
}

// Handle model change
async function handleModelChange() {
    const modelType = elements.modelSelect.value;
    if (!modelType) return;

    showLoading(true);
    updateStatus(elements.modelStatus, 'Loading...', 'loading');

    try {
        // Load model on server
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: modelType }),
        });

        if (!response.ok) {
            throw new Error('Failed to load model on server');
        }

        // Load ONNX model in browser
        await loadONNXModel(modelType);

        state.currentModel = modelType;
        updateStatus(elements.modelStatus, `${modelType} loaded`, 'success');
        elements.imageUpload.disabled = false;

        showToast('Model loaded successfully', 'success');
    } catch (e) {
        console.error('Model load error:', e);
        updateStatus(elements.modelStatus, 'Failed', 'error');
        showToast(`Failed to load model: ${e.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Load ONNX model for browser inference
async function loadONNXModel(modelType) {
    try {
        const modelPath = `/models/onnx/sam_decoder_${modelType}.onnx`;

        // Configure ONNX Runtime with WebGPU fallback to WASM
        const options = {
            executionProviders: ['webgpu', 'wasm'],
        };

        state.onnxSession = await ort.InferenceSession.create(modelPath, options);
        console.log('ONNX model loaded:', modelType);
    } catch (e) {
        console.error('ONNX load error:', e);
        throw e;
    }
}

// Handle image upload
async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!state.currentModel) {
        showToast('Please select a model first', 'error');
        return;
    }

    showLoading(true);
    updateStatus(elements.uploadStatus, 'Uploading...', 'loading');

    try {
        // Upload image to server
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload_image', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to upload image');
        }

        const data = await response.json();
        state.imageId = data.image_id;

        // Load image for display
        const reader = new FileReader();
        reader.onload = async (event) => {
            const img = new Image();
            img.onload = async () => {
                state.currentImage = img;
                initCanvas(img);

                // Load embedding
                await loadEmbedding(data.embedding_url);

                // Reset state
                state.clicks = [];
                state.currentMask = null;
                updateUI();

                updateStatus(elements.uploadStatus, 'Ready', 'success');
                elements.imageInfo.textContent = `${data.width} × ${data.height} px`;

                elements.undoBtn.disabled = false;
                elements.clearBtn.disabled = false;
                elements.exportBtn.disabled = false;
                elements.dilationSlider.disabled = false;

                showToast('Image loaded successfully', 'success');
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    } catch (e) {
        console.error('Upload error:', e);
        updateStatus(elements.uploadStatus, 'Failed', 'error');
        showToast(`Failed to upload image: ${e.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Load embedding from server
async function loadEmbedding(embeddingUrl) {
    try {
        const response = await fetch(embeddingUrl);
        if (!response.ok) {
            throw new Error('Failed to load embedding');
        }

        const arrayBuffer = await response.arrayBuffer();

        // Parse .npy file
        state.imageEmbedding = parseNpy(arrayBuffer);
        console.log('Embedding loaded:', state.imageEmbedding.shape);
    } catch (e) {
        console.error('Embedding load error:', e);
        throw e;
    }
}

// Parse .npy file format
function parseNpy(arrayBuffer) {
    const view = new DataView(arrayBuffer);

    // Skip magic string and version
    let offset = 8;

    // Read header length
    const headerLen = view.getUint16(offset, true);
    offset += 2;

    // Read header (JSON-like dict)
    const headerBytes = new Uint8Array(arrayBuffer, offset, headerLen);
    const headerStr = new TextDecoder().decode(headerBytes);
    offset += headerLen;

    // Parse shape and dtype from header
    const shapeMatch = headerStr.match(/'shape':\s*\(([^)]+)\)/);
    const dtypeMatch = headerStr.match(/'descr':\s*'([^']+)'/);

    const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    const dtype = dtypeMatch[1];

    // Read data
    let data;
    if (dtype.includes('f4')) {
        data = new Float32Array(arrayBuffer, offset);
    } else if (dtype.includes('f8')) {
        data = new Float64Array(arrayBuffer, offset);
    } else {
        throw new Error('Unsupported dtype: ' + dtype);
    }

    return { data, shape, dtype };
}

// Initialize canvas
function initCanvas(img) {
    state.canvas.width = img.width;
    state.canvas.height = img.height;
    state.scale = 1;
    state.offsetX = 0;
    state.offsetY = 0;

    drawCanvas();
}

// Draw canvas with image, mask, and points
function drawCanvas() {
    const ctx = state.ctx;
    const canvas = state.canvas;

    // Clear
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply transform
    ctx.setTransform(state.scale, 0, 0, state.scale, state.offsetX, state.offsetY);

    // Draw image
    if (state.currentImage) {
        ctx.drawImage(state.currentImage, 0, 0);
    }

    // Draw mask overlay
    if (state.currentMask) {
        ctx.globalAlpha = 0.5;
        ctx.drawImage(state.currentMask, 0, 0);
        ctx.globalAlpha = 1.0;
    }

    // Draw clicks
    state.clicks.forEach(click => {
        ctx.beginPath();
        ctx.arc(click.x, click.y, 5 / state.scale, 0, 2 * Math.PI);
        ctx.strokeStyle = '#3182ce';
        ctx.lineWidth = 3 / state.scale;
        ctx.stroke();

        if (click.label === 1) {
            // Positive: filled
            ctx.fillStyle = '#3182ce';
            ctx.fill();
        }
        // Negative: hollow (already stroked)
    });
}

// Handle mouse down
function handleMouseDown(e) {
    const rect = state.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert to image coordinates
    const imgX = (x - state.offsetX) / state.scale;
    const imgY = (y - state.offsetY) / state.scale;

    if (e.button === 0) {
        // Left click: add positive point
        if (imgX >= 0 && imgX < state.currentImage.width && imgY >= 0 && imgY < state.currentImage.height) {
            addClick(imgX, imgY, 1);
            showPointFeedback(e.clientX, e.clientY);
        }
    } else if (e.button === 2) {
        // Right click: add negative point
        if (imgX >= 0 && imgX < state.currentImage.width && imgY >= 0 && imgY < state.currentImage.height) {
            addClick(imgX, imgY, 0);
            showPointFeedback(e.clientX, e.clientY);
        }
    } else if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
        // Middle click or shift+left: start dragging
        state.isDragging = true;
        state.lastX = x;
        state.lastY = y;
        state.canvas.style.cursor = 'grabbing';
    }
}

// Handle mouse move
function handleMouseMove(e) {
    if (state.isDragging) {
        const rect = state.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const dx = x - state.lastX;
        const dy = y - state.lastY;

        state.offsetX += dx;
        state.offsetY += dy;

        state.lastX = x;
        state.lastY = y;

        drawCanvas();
    }
}

// Handle mouse up
function handleMouseUp(e) {
    if (state.isDragging) {
        state.isDragging = false;
        state.canvas.style.cursor = 'crosshair';
    }
}

// Handle wheel zoom
function handleWheel(e) {
    e.preventDefault();

    const rect = state.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Zoom factor
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = state.scale * delta;

    // Limit scale
    if (newScale < 0.1 || newScale > 10) return;

    // Zoom towards cursor
    state.offsetX = x - (x - state.offsetX) * delta;
    state.offsetY = y - (y - state.offsetY) * delta;
    state.scale = newScale;

    drawCanvas();
}

// Add click and run inference
function addClick(x, y, label) {
    if (!state.onnxSession || !state.imageEmbedding) {
        showToast('Model not ready', 'error');
        return;
    }

    state.clicks.push({ x, y, label });
    updateUI();

    // Debounced inference
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
        runInference();
    }, 100);
}

// Run ONNX inference
async function runInference() {
    if (state.clicks.length === 0) {
        state.currentMask = null;
        drawCanvas();
        return;
    }

    try {
        // Prepare inputs
        const imageEmbeddings = new ort.Tensor(
            'float32',
            state.imageEmbedding.data,
            state.imageEmbedding.shape
        );

        // Point coords: [batch, num_points, 2]
        const numPoints = state.clicks.length;
        const pointCoords = new Float32Array(numPoints * 2);
        const pointLabels = new Float32Array(numPoints);

        for (let i = 0; i < numPoints; i++) {
            pointCoords[i * 2] = state.clicks[i].x;
            pointCoords[i * 2 + 1] = state.clicks[i].y;
            pointLabels[i] = state.clicks[i].label;
        }

        const pointCoordsTensor = new ort.Tensor('float32', pointCoords, [1, numPoints, 2]);
        const pointLabelsTensor = new ort.Tensor('float32', pointLabels, [1, numPoints]);

        // Mask input: empty for first prediction
        const maskInput = new Float32Array(256 * 256).fill(0);
        const maskInputTensor = new ort.Tensor('float32', maskInput, [1, 1, 256, 256]);

        const hasMaskInput = new ort.Tensor('float32', new Float32Array([0]), [1]);

        // Original image size
        const origImSize = new ort.Tensor(
            'float32',
            new Float32Array([state.currentImage.height, state.currentImage.width]),
            [2]
        );

        // Run inference
        const feeds = {
            image_embeddings: imageEmbeddings,
            point_coords: pointCoordsTensor,
            point_labels: pointLabelsTensor,
            mask_input: maskInputTensor,
            has_mask_input: hasMaskInput,
            orig_im_size: origImSize,
        };

        const results = await state.onnxSession.run(feeds);

        // Get mask
        const masks = results.masks;
        const maskData = masks.data;
        const maskShape = masks.dims; // [batch, num_masks, height, width]

        // Convert mask to canvas
        const maskCanvas = createMaskCanvas(
            maskData,
            maskShape[2],
            maskShape[3],
            state.dilation
        );

        state.currentMask = maskCanvas;

        // Calculate area
        const area = calculateMaskArea(maskData, maskShape[2], maskShape[3]);
        elements.maskArea.textContent = `${area.toLocaleString()} px²`;

        drawCanvas();
    } catch (e) {
        console.error('Inference error:', e);
        showToast('Inference failed', 'error');
    }
}

// Create mask canvas with optional dilation
function createMaskCanvas(maskData, height, width, dilation) {
    // Convert to binary mask
    const binaryMask = new Uint8ClampedArray(height * width);
    for (let i = 0; i < height * width; i++) {
        binaryMask[i] = maskData[i] > 0 ? 255 : 0;
    }

    // Apply dilation if needed
    let processedMask = binaryMask;
    if (dilation > 0) {
        processedMask = dilateMask(binaryMask, width, height, dilation);
    }

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Create blue overlay
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < height * width; i++) {
        const val = processedMask[i];
        imageData.data[i * 4] = 49;      // R
        imageData.data[i * 4 + 1] = 130; // G
        imageData.data[i * 4 + 2] = 206; // B
        imageData.data[i * 4 + 3] = val; // A
    }

    ctx.putImageData(imageData, 0, 0);

    // Draw contours
    drawContours(ctx, processedMask, width, height);

    return canvas;
}

// Simple morphological dilation (JavaScript implementation)
function dilateMask(mask, width, height, radius) {
    const result = new Uint8ClampedArray(width * height);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let maxVal = 0;

            // Check neighborhood
            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    // Circular kernel
                    if (dx * dx + dy * dy > radius * radius) continue;

                    const nx = x + dx;
                    const ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const idx = ny * width + nx;
                        maxVal = Math.max(maxVal, mask[idx]);
                    }
                }
            }

            result[y * width + x] = maxVal;
        }
    }

    return result;
}

// Draw contours on canvas
function drawContours(ctx, mask, width, height) {
    ctx.strokeStyle = '#3182ce';
    ctx.lineWidth = 2;

    // Simple edge detection
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;

            if (mask[idx] > 0) {
                // Check if on edge
                if (mask[idx - 1] === 0 || mask[idx + 1] === 0 ||
                    mask[idx - width] === 0 || mask[idx + width] === 0) {
                    ctx.fillStyle = '#3182ce';
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        }
    }
}

// Calculate mask area
function calculateMaskArea(maskData, height, width) {
    let count = 0;
    for (let i = 0; i < height * width; i++) {
        if (maskData[i] > 0) count++;
    }
    return count;
}

// Handle undo
function handleUndo() {
    if (state.clicks.length > 0) {
        state.clicks.pop();
        updateUI();
        runInference();
    }
}

// Handle clear
function handleClear() {
    state.clicks = [];
    state.currentMask = null;
    updateUI();
    drawCanvas();
}

// Handle dilation change
function handleDilationChange(e) {
    state.dilation = parseInt(e.target.value);
    elements.dilationValue.textContent = state.dilation;

    // Re-run inference with new dilation
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
        runInference();
    }, 100);
}

// Handle export
async function handleExport() {
    if (!state.imageId || state.clicks.length === 0) {
        showToast('No data to export', 'error');
        return;
    }

    try {
        // Extract contours from current mask
        const contours = extractContoursFromMask(state.currentMask);

        const exportData = {
            clicks: state.clicks,
            dilation: state.dilation,
            contours: contours,
        };

        const response = await fetch(`/api/export/${state.imageId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(exportData),
        });

        if (!response.ok) {
            throw new Error('Export failed');
        }

        const data = await response.json();

        // Download JSON
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sam_export_${state.imageId}.json`;
        a.click();
        URL.revokeObjectURL(url);

        showToast('Exported successfully', 'success');
    } catch (e) {
        console.error('Export error:', e);
        showToast('Export failed', 'error');
    }
}

// Extract contours from mask canvas
function extractContoursFromMask(maskCanvas) {
    if (!maskCanvas) return [];

    const ctx = maskCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    // Convert to binary
    const binary = new Uint8Array(maskCanvas.width * maskCanvas.height);
    for (let i = 0; i < binary.length; i++) {
        binary[i] = data[i * 4 + 3] > 128 ? 1 : 0;
    }

    // Simple contour extraction (boundary points)
    const contours = [];
    const visited = new Set();

    for (let y = 0; y < maskCanvas.height; y++) {
        for (let x = 0; x < maskCanvas.width; x++) {
            const idx = y * maskCanvas.width + x;
            if (binary[idx] === 1 && !visited.has(idx)) {
                const contour = traceContour(binary, maskCanvas.width, maskCanvas.height, x, y, visited);
                if (contour.length > 0) {
                    contours.push(contour);
                }
            }
        }
    }

    return contours;
}

// Trace contour from a starting point
function traceContour(binary, width, height, startX, startY, visited) {
    const contour = [];
    const stack = [[startX, startY]];

    while (stack.length > 0 && contour.length < 10000) {
        const [x, y] = stack.pop();
        const idx = y * width + x;

        if (visited.has(idx)) continue;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        if (binary[idx] === 0) continue;

        visited.add(idx);

        // Check if boundary point
        const isBoundary = (
            x === 0 || x === width - 1 || y === 0 || y === height - 1 ||
            binary[idx - 1] === 0 || binary[idx + 1] === 0 ||
            binary[idx - width] === 0 || binary[idx + width] === 0
        );

        if (isBoundary) {
            contour.push([x, y]);
        }

        // Add neighbors (limit growth)
        if (contour.length < 5000) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if (dx === 0 && dy === 0) continue;
                    stack.push([x + dx, y + dy]);
                }
            }
        }
    }

    return contour;
}

// Show point feedback animation
function showPointFeedback(x, y) {
    const feedback = document.createElement('div');
    feedback.className = 'point-feedback';
    feedback.style.left = `${x - 10}px`;
    feedback.style.top = `${y - 10}px`;
    document.body.appendChild(feedback);

    setTimeout(() => {
        feedback.remove();
    }, 500);
}

// Update UI
function updateUI() {
    elements.clickCount.textContent = state.clicks.length;
}

// Show toast notification
function showToast(message, type = 'info') {
    elements.toast.textContent = message;
    elements.toast.className = `toast ${type} show`;

    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// Show/hide loading overlay
function showLoading(show) {
    if (show) {
        elements.loadingOverlay.classList.add('active');
    } else {
        elements.loadingOverlay.classList.remove('active');
    }
}

// Update status badge
function updateStatus(element, text, type) {
    element.textContent = text;
    element.className = `status-badge ${type}`;
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
