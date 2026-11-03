import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Projects/AIBSI";
const SKILL_DIR = "C:/Users/ASUS/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations";
const RUNTIME_PYTHON = "C:/Users/ASUS/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const sourceTemplate = path.join(workspaceDir, "presentation_progress/output/NeuroTensor_Status_Presentation_10_Slides_EEG_Pipeline.pptx");
const buildDir = path.join(workspaceDir, "presentation_progress/.codex_build/tensor_progress");
const renderDir = path.join(buildDir, "render");
const finalPath = path.join(workspaceDir, "presentation_progress/output/NeuroTensor_Progress_Tensor_Hybrid_2026_Final.pptx");

const COLORS = {
  navy: "#102B45",
  dark: "#12304A",
  teal: "#008487",
  blue: "#4F7DB7",
  orange: "#DD8230",
  gray: "#566779",
  pale: "#CEE2E8",
  light: "#E8EFF3",
  white: "#FFFFFF",
};

const spectralImage = path.join(workspaceDir, "neural_comparison/advanced_clean_spectral/outputs/results/spectral_examples.png");
const rankImage = path.join(workspaceDir, "neural_comparison/advanced_clean_spectral/outputs/tensor_train_svm/tt_rank_selection.png");

function setText(presentation, id, value, style = undefined) {
  const shape = presentation.resolve(id);
  shape.text = value;
  if (style) shape.text.style = { ...shape.text.style, ...style };
  return shape;
}

function addText(slide, text, position, style) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { typeface: "Arial", ...style };
  return shape;
}

function addRule(slide, left, top, width, height = 2, color = COLORS.light) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function setNotes(slide, lines) {
  slide.speakerNotes.text = lines.join("\n");
}

async function writeBlob(target, blob) {
  await fs.writeFile(target, new Uint8Array(await blob.arrayBuffer()));
}

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(renderDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourceTemplate));
const spectralBytes = await fs.readFile(spectralImage);
const rankBytes = await fs.readFile(rankImage);

// Slide 1: cover
setText(presentation, "sh/fu94fe98", "Project progress");
setText(presentation, "sh/547294r6", "Tensor representation, spectrograms and hybrid EEG classification");
setText(presentation, "sh/7qp4be9c", "Resting EEG classification of\nAlzheimer’s disease, frontotemporal\ndementia and healthy controls");
setText(presentation, "sh/65g3298r", "Progress from handcrafted EEG features to\nTensor Train compression and EEGPT fusion");
setText(presentation, "sh/sryl4zqx", "Updated status presentation · October 2026");
setNotes(presentation.slides.getItem(0), [
  "This presentation summarizes the experiments completed after the first status review.",
  "The main additions are the combined cohort, log-mel inputs, Tensor Train decomposition, pretrained EEGPT and participant-level stability testing.",
]);

// Slide 2: research origin
setText(presentation, "sh/mtwrmxg7", "Research origin");
setText(presentation, "sh/9072xkry", "The initial hypothesis came from Miltiadous et al. and the AHEPA EEG programme");
setText(presentation, "sh/b29kza94", "Starting paper");
setText(presentation, "sh/a10jqpsj", "Miltiadous et al. (2021) tested whether quantitative EEG features could distinguish AD, FTD and healthy controls.");
setText(presentation, "sh/x4r21kru", "Public dataset");
setText(presentation, "sh/w3i1sfa9", "The 2023 descriptor released eyes-closed recordings from 88 people: 36 AD, 23 FTD and 29 controls, using 19 electrodes.");
setText(presentation, "sh/r65knqtk", "Core methodological lesson");
setText(presentation, "sh/q5wjelsz", "Accuracy drops when the test set contains unseen people. Participant-level validation matters more than epoch-level scores.");
setText(presentation, "sh/9wnqhczy", "Miltiadous et al., Diagnostics 2021, 11, 1437 · Data 2023, 8, 95");
setNotes(presentation.slides.getItem(1), [
  "Sources:",
  "https://doi.org/10.3390/diagnostics11081437",
  "https://doi.org/10.3390/data8060095",
  "https://openneuro.org/datasets/ds004504",
]);

// Slide 3: their method versus ours
setText(presentation, "sh/m1c3mlsn", "Initial approach and our extension");
setText(presentation, "sh/cza94vmx", "We kept the clinical question but changed the data scale, representation and evaluation");
setText(presentation, "sh/298ryl4v", "Their approach");
setText(presentation, "sh/3ah8rqlg", "A small 28-person cohort, handcrafted statistical and spectral features, six conventional classifiers and three binary comparisons.");
setText(presentation, "sh/obq90bml", "Their validation finding");
setText(presentation, "sh/pcjqtg36", "Epoch-level 10-fold accuracy reached 96–99%, while leave-one-patient-out results fell to 60–86% depending on the task and classifier.");
setText(presentation, "sh/m5cra54z", "Our research question");
setText(presentation, "sh/n6ls3alk", "Can a single three-class, participant-level model improve when spectral summaries, tensor compression and pretrained raw-EEG features are compared fairly?");
setText(presentation, "sh/n2l4fq98", "Main change: AD, FTD and healthy controls are predicted together for each participant");
setNotes(presentation.slides.getItem(2), [
  "The original work reported AD versus control, FTD versus control and AD versus FTD separately.",
  "Their leave-one-patient-out table reported 78.5% for AD versus control with decision trees and 86.3% for FTD versus control with random forests.",
  "Source: https://doi.org/10.3390/diagnostics11081437",
]);

// Slide 4: data progress
setText(presentation, "sh/kbm987y5", "Dataset expansion and preparation");
setText(presentation, "sh/1cj2d8b6", "The combined experiment now includes 174 participants from three acquisition sites");
setText(presentation, "sh/rm1k7yt4", "AHEPA / ds004504");
setText(presentation, "sh/ql8jytsj", "88 participants\n36 AD · 23 FTD · 29 healthy\n19 electrodes · eyes closed · 500 Hz");
setText(presentation, "sh/doj29oba", "BrainLat cohort");
setText(presentation, "sh/sna103ap", "86 participants from AR and CL\n35 AD · 19 FTD · 32 healthy\nMapped to the same 19-electrode montage");
setText(presentation, "sh/3ihk3et8", "Combined cohort: 71 AD · 42 FTD · 61 healthy controls");
setText(presentation, "sh/ih8ju9sn", "All windows from one participant remain in the same training, validation or test fold.");
setNotes(presentation.slides.getItem(3), [
  "Combined manifest: 174 participants across AHEPA, AR and CL.",
  "The preparation pipeline harmonizes channel names, references, sampling rates and labels before feature extraction.",
]);

// Slide 5: updated pipeline, reusing the original template illustration
setText(presentation, "sh/1cfmhgne", "Updated EEG to tensor and hybrid features");
setText(presentation, "sh/0b65obm9", "The current pipeline keeps both spectral structure and the raw temporal signal");
setText(presentation, "sh/nex4jq5k", "Moderately cleaned EEG");
setText(presentation, "sh/mdonql4z", "19 common electrodes\n1–40 Hz filtering");
setText(presentation, "sh/9gzml0nq", "Four-second windows");
setText(presentation, "sh/ofq5svm5", "Participant-safe segmentation\nartifact rejection");
setText(presentation, "sh/bih4na5w", "Log-mel spectrum");
setText(presentation, "sh/ah8nu54b", "19 × 24 × 13\nelectrode × mel × time");
setText(presentation, "sh/oza1gfyh", "Participant tensor");
setText(presentation, "sh/p0ji9kf2", "19 × 24 × 4\nmean · SD · median · IQR");
setText(presentation, "sh/2xsjepgb", "Two model branches");
setText(presentation, "sh/3y107axw", "Tensor / SVM features\nraw EEG for EEGPT");
setText(presentation, "sh/cnu1kzyd", "Tensor bases, site normalization and model settings are fitted using training participants only.");
setNotes(presentation.slides.getItem(4), [
  "The illustration is retained from the earlier deck, while the labels now describe the current pipeline.",
  "The SVM branch summarizes log-mel windows. EEGPT receives raw 19-channel four-second windows resampled to 256 Hz.",
]);

// Slide 6: spectrogram examples and exact tensor shapes
setText(presentation, "sh/ahkvi1cb", "Spectrograms and tensor construction");
setText(presentation, "sh/dgbulwnm", "STFT preserves linear frequency detail; log-mel gives a compact frequency representation");
presentation.delete(["sh/z2tcnm5s", "sh/yhkbe1o7", "sh/l4bupwny", "sh/032tgr6d", "sh/n6dcr65o", "sh/m5kbi1oj", "im/zu9wb6pc"]);
const slide6 = presentation.slides.getItem(5);
slide6.images.add({
  blob: spectralBytes,
  contentType: "image/png",
  alt: "Example log-STFT and log-mel spectrograms from channel C3",
  fit: "contain",
  position: { left: 54, top: 190, width: 770, height: 390 },
});
addRule(slide6, 842, 194, 2, 400, COLORS.light);
addText(slide6, "Window tensor", { left: 870, top: 205, width: 310, height: 42 }, { fontSize: 28, bold: true, color: COLORS.teal });
addText(slide6, "19 × 24 × 13", { left: 870, top: 250, width: 310, height: 54 }, { fontSize: 36, bold: true, color: COLORS.navy });
addText(slide6, "electrodes × mel bands × time frames", { left: 870, top: 307, width: 310, height: 58 }, { fontSize: 21, color: COLORS.gray });
addText(slide6, "Participant tensor", { left: 870, top: 389, width: 310, height: 42 }, { fontSize: 28, bold: true, color: COLORS.teal });
addText(slide6, "19 × 24 × 4", { left: 870, top: 434, width: 310, height: 54 }, { fontSize: 36, bold: true, color: COLORS.navy });
addText(slide6, "mean\nstandard deviation\nmedian\ninterquartile range", { left: 870, top: 495, width: 310, height: 110 }, { fontSize: 21, color: COLORS.gray });
setText(presentation, "sh/x4vedgvm", "The displayed C3 example spans 8 seconds; model inputs use four-second windows with 13 log-mel frames.");
setNotes(slide6, [
  "The SVM statistical summary uses mean and standard deviation, producing 19 × 48 = 912 participant features before PCA.",
  "The Tensor Train experiment also retains median and interquartile range, producing 19 × 24 × 4 = 1,824 values per participant.",
  `Figure source: ${spectralImage}`,
]);

// Slide 7: Tensor Train method
setText(presentation, "sh/w32twb6l", "Tensor Train decomposition");
setText(presentation, "sh/yhg7epsj", "A training-fitted basis compresses the participant × electrode × mel × statistic tensor");
presentation.delete(["sh/87ipkzal", "sh/98rqt4r6", "sh/ml07i9sv", "sh/7m98ru9g", "sh/oryp8fah", "sh/pc76hkr2"]);
const slide7 = presentation.slides.getItem(6);
addText(slide7, "Input", { left: 64, top: 205, width: 510, height: 42 }, { fontSize: 29, bold: true, color: COLORS.teal });
addText(slide7, "174 × 19 × 24 × 4", { left: 64, top: 248, width: 510, height: 50 }, { fontSize: 35, bold: true, color: COLORS.navy });
addText(slide7, "The observation mode follows the feature modes during TT-SVD, so validation participants can be contracted against the same basis.", { left: 64, top: 304, width: 510, height: 115 }, { fontSize: 23, color: COLORS.gray });
addText(slide7, "Selected compression", { left: 64, top: 441, width: 510, height: 42 }, { fontSize: 29, bold: true, color: COLORS.teal });
addText(slide7, "Terminal rank 4 produced four participant features. It retained a compact representation but had 63.7% centered reconstruction error.", { left: 64, top: 486, width: 510, height: 105 }, { fontSize: 23, color: COLORS.gray });
slide7.images.add({
  blob: rankBytes,
  contentType: "image/png",
  alt: "Development macro F1 for Tensor Train terminal ranks 4, 8, 16 and 32",
  fit: "contain",
  position: { left: 610, top: 195, width: 590, height: 410 },
});
setText(presentation, "sh/h4bupgn6", "Rank and SVM settings were selected inside development folds; the held-out test labels were not used.");
setNotes(slide7, [
  "Training-only TT basis fitting prevents independently decomposed participants from having incompatible signs or rotations.",
  "Selected ranks: channel rank 4, spectral rank 24 and terminal rank 4.",
  `Figure source: ${rankImage}`,
]);

// Slide 8: improvements made
setText(presentation, "sh/l0vuh0rm", "What changed to improve performance");
setText(presentation, "sh/cb2tkvap", "The largest gains came from data harmonization and complementary representations");
presentation.delete("ch/zuhc7edk");
const slide8 = presentation.slides.getItem(7);
addText(slide8, "1  Data and cleaning", { left: 64, top: 213, width: 350, height: 46 }, { fontSize: 29, bold: true, color: COLORS.teal });
addText(slide8, "Combined two cohorts\nStandardized the 19-channel montage\nUsed moderate artifact rejection\nAvoided excessive interpolation", { left: 64, top: 273, width: 350, height: 170 }, { fontSize: 23, color: COLORS.navy });
addRule(slide8, 426, 205, 2, 330, COLORS.light);
addText(slide8, "2  Representation", { left: 456, top: 213, width: 350, height: 46 }, { fontSize: 29, bold: true, color: COLORS.teal });
addText(slide8, "Log-mel summaries for SVM\nRaw EEG for pretrained EEGPT\nTensor Train as a course ablation\nProbability fusion of both branches", { left: 456, top: 273, width: 350, height: 170 }, { fontSize: 23, color: COLORS.navy });
addRule(slide8, 818, 205, 2, 330, COLORS.light);
addText(slide8, "3  Evaluation", { left: 848, top: 213, width: 350, height: 46 }, { fontSize: 29, bold: true, color: COLORS.teal });
addText(slide8, "Participant-level splits\nTraining-only site normalization\nMacro F1 and class recall\nRepeated nested cross-validation", { left: 848, top: 273, width: 350, height: 170 }, { fontSize: 23, color: COLORS.navy });
setText(presentation, "sh/zedcfa9g", "SVM + EEGPT fusion improved the fixed split from 66.7% to 70.4% accuracy");
setText(presentation, "sh/kzmdova1", "The repeated experiments show that FTD sample size and site shift still limit generalization.");
setNotes(slide8, [
  "Advanced automatic cleaning interpolated channels in 114 of 174 participants and reduced macro F1, so the moderate pipeline was retained.",
  "The blend combines calibrated probability outputs from the spectral SVM and fine-tuned EEGPT.",
]);

// Slide 9: final editable result chart
setText(presentation, "sh/sjix0zy1", "Model results and tensor ablation");
setText(presentation, "sh/18byd4zy", "Exploratory comparison on the same fixed 27-participant test split");
presentation.delete("ch/o72p4vml");
const slide9 = presentation.slides.getItem(8);
const chart = slide9.charts.add("bar", {
  position: { left: 64, top: 205, width: 1152, height: 385 },
  categories: ["Spectral SVM", "EEGPT", "Random Forest", "TT + SVM", "SVM + EEGPT"],
  series: [
    { name: "Accuracy", values: [0.667, 0.630, 0.556, 0.519, 0.704], fill: COLORS.teal, valuesFormatCode: "0%" },
    { name: "Macro F1", values: [0.643, 0.550, 0.489, 0.464, 0.678], fill: COLORS.blue, valuesFormatCode: "0%" },
    { name: "Macro recall", values: [0.642, 0.565, 0.504, 0.468, 0.676], fill: COLORS.orange, valuesFormatCode: "0%" },
  ],
  hasLegend: true,
  legend: { position: "bottom" },
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 85 },
  dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 12, color: COLORS.navy } },
  yAxis: {
    minimumScale: 0,
    maximumScale: 1,
    majorUnit: 0.2,
    numberFormatCode: "0%",
    majorGridlines: { style: "solid", fill: "#D7E0E7", width: 1 },
  },
  xAxis: { tickLabelPosition: "low" },
  chartFill: "#FFFFFF",
  chartLine: { style: "solid", fill: "none", width: 0 },
  plotAreaFill: "#FFFFFF",
  plotAreaLine: { style: "solid", fill: "none", width: 0 },
});
const { applyPresentationChartFont } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
applyPresentationChartFont(chart, { fontFamily: "Arial" });
setText(presentation, "sh/6h0fypgb", "Tensor Train preserved structure but removed useful class information");
setText(presentation, "sh/tkby9kzm", "Best fixed result: 70.4% accuracy, 67.8% macro F1 and 67.6% macro recall.");
setNotes(slide9, [
  "Fixed test set: 11 AD, 6 FTD and 10 healthy controls.",
  "Tensor Train + SVM: 51.9% accuracy and 46.4% macro F1.",
  "SVM + EEGPT: 70.4% accuracy and 67.8% macro F1.",
  "These are fixed-split exploratory results. Repeated participant-level estimates remain lower and more variable.",
]);

// Slide 10: wrap-up
setText(presentation, "sh/s7yt4b21", "Current conclusion and final benchmark");
setText(presentation, "sh/xc3mho32", "The project now includes a complete tensor ablation and a stronger hybrid EEG model");
setText(presentation, "sh/zelojyl8", "Completed");
setText(presentation, "sh/ydcnatkn", "174-participant harmonized dataset\nSTFT and log-mel representations\nTensor Train decomposition with SVM\nCNN, EEGPT, SVM and Random Forest comparisons");
setText(presentation, "sh/l036l83y", "Final project benchmark");
setText(presentation, "sh/kzu5c32t", "SVM + EEGPT probability fusion\n70.4% fixed-split accuracy\n67.8% fixed-split macro F1\nTensor Train retained as the tensor-course ablation");
setText(presentation, "sh/m1cnetkj", "Main limitation: only 42 FTD participants and substantial performance variation across sites and participant splits.");
setNotes(presentation.slides.getItem(9), [
  "Present the SVM + EEGPT result as the strongest fixed-split benchmark, not as external clinical accuracy.",
  "The tensor contribution is the training-fitted TT representation and the finding that compression did not improve diagnostic separation.",
]);

// Export previews and validate.
for (const [index, slide] of presentation.slides.items.entries()) {
  const name = `slide-${String(index + 1).padStart(2, "0")}.png`;
  await writeBlob(path.join(renderDir, name), await presentation.export({ slide, format: "png", scale: 1 }));
}
await writeBlob(path.join(renderDir, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 0.5 }));

const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const candidatePath = path.join(buildDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [9],
  materializeLiteralChartWorkbooks: true,
  sourceTemplatePath: sourceTemplate,
  fontPolicy: {
    basis: "reference",
    families: ["Arial"],
    referencePath: sourceTemplate,
    referenceSha256: "31a549baac9d64dfc6419b728c2a6f7b737e6d8c7c02d81a02c7601e4d7589e8",
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "validation-final.json"),
});

console.log(JSON.stringify({ finalPath, result }, null, 2));
