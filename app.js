import { MgpstrForSceneTextRecognition, MgpstrProcessor, RawImage } from '@ro7m/ttransform.js';

let model;
let processor;

async function initializeModel() {
    const model_id = 'onnx-community/mgp-str-base';
    model = await MgpstrForSceneTextRecognition.from_pretrained(model_id);
    processor = await MgpstrProcessor.from_pretrained(model_id);
}

async function recognizeText(image) {
    // Show spinner
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('result').innerText = '';

    // Preprocess the image
    const result = await processor(image);

    // Perform inference
    const outputs = await model(result);

    // Decode the model outputs
    const generated_text = processor.batch_decode(outputs.logits).generated_text;

    // Hide spinner and display the result
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('result').innerText = `Recognized Text: ${generated_text[0]}`;
}

document.getElementById('imageUpload').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const imageUrl = URL.createObjectURL(file);
        const image = await RawImage.read(imageUrl);
        recognizeText(image);
    }
});

// Initialize the model when the page loads
window.addEventListener('load', initializeModel);
