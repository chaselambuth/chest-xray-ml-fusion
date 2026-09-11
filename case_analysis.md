# **Case Analysis: What the X-ray Fusion Notebooks Discovered**

# **Purpose**

This case analysis summarizes the experiment trail from the six X-ray fusion notebooks. The notebooks already contain metrics, model comparisons, Grad-CAM overlays, and tuning artifacts. 

The analysis uses the held-out test split and the ordered notebook artifacts:

- Baseline setup: experiment_notebooks/01_Xray_Baseline.ipynb
- Text encoder selection: experiment_notebooks/02_Xray_Text_Encoder_Selection.ipynb
- Residual gated fusion: experiment_notebooks/03_Xray_ClinicalBERT_Gated_Fusion_Test.ipynb
- Segmentation/preprocessing ablation: experiment_notebooks/04_Xray_Segmentation_Test.ipynb
- Hyperparameter and augmentation tuning: experiment_notebooks/05_Xray_Hyperparameter_Augmentation.ipynb
- Final Grad-CAM interpretability: experiment_notebooks/06_Xray_GradCAM_Interpretability.ipynb



# **Executive Summary**

The project discovered that the strongest final direction is a Bio_ClinicalBERT residual gated fusion model using the original full frontal and lateral X-ray images. The model benefits from image/text fusion and from a gated fusion block, but it remains limited by heavy class imbalance and by unstable minority-disease recognition.

The final model is good at recognizing normal studies and sometimes produces visually plausible Grad-CAM attention for abnormal cases. However, the dominant failure mode is still abnormal studies being predicted as normal. On the held-out test Grad-CAM sweep, 79 of 124 errors were abnormal-to-normal misses. That means about 64% of all test errors were not random confusion among disease classes; they were missed abnormal findings absorbed into the normal class.

The notebooks also discovered that not every apparent model error is cleanly a model error. Several case rows reveal label ambiguity or label noise, especially when the compact five-class target disagrees with details in the MeSH tags, findings, or impression text. This matters because the model is being judged against a simplified study label derived from messy radiology metadata.

# **Dataset And Split Context**

The labeled dataset used by the notebooks contains 3,144 studies after filtering to usable frontal/lateral pairs and assigning compact five-class labels.


| Split      | Total | Normal | Pneumonia/Opacity | Cardiomegaly | Pleural Effusion | Chronic Lung Disease |
| ---------- | ----- | ------ | ----------------- | ------------ | ---------------- | -------------------- |
| Train      | 2,200 | 1,611  | 216               | 178          | 120              | 75                   |
| Validation | 472   | 346    | 47                | 38           | 25               | 16                   |
| Test       | 472   | 345    | 46                | 38           | 26               | 17                   |
| Total      | 3,144 | 2,302  | 309               | 254          | 171              | 108                  |


The most important dataset fact is the imbalance: normal studies are about 73% of the labeled set and about 73% of the held-out test set. Every notebook result should be read through that lens. Accuracy can look acceptable while the model still performs weakly on minority disease classes.

# **What Each Notebook Discovered**



## Notebook 01: Baselines Established The Problem

Notebook 01 created the five-class study-level dataset and trained the first baselines.

Key findings:

- The text-only baseline was too weak to act as the main diagnostic signal, reaching 0.396 test accuracy and 0.242 macro F1.
- The dual-view image model was a much stronger starting point, reaching 0.750 test accuracy.
- The first multimodal fusion model had the best baseline accuracy at 0.769 and improved macro F1 to 0.368.
- Even the first fusion model leaned heavily on the normal class. Normal recall was high, but abnormal recall remained low.

Discovery: image information is essential, text helps, but simple fusion does not solve class imbalance.

### Notebook 02: Text Encoder Selection Was Not A Clean Sweep

Notebook 02 compared CheXbert, Bio_ClinicalBERT, and RadBERT under aligned text-only and fusion settings.


| Model              | Task       | Test Accuracy | Test Macro F1 |
| ------------------ | ---------- | ------------- | ------------- |
| Bio_ClinicalBERT   | fusion     | 0.763         | 0.311         |
| CheXbert           | fusion     | 0.750         | 0.404         |
| RadBERT            | fusion     | 0.746         | 0.347         |
| Dual-view ResNet18 | image only | 0.756         | 0.360         |
| CheXbert           | text only  | 0.557         | 0.207         |
| Bio_ClinicalBERT   | text only  | 0.502         | 0.203         |
| RadBERT            | text only  | 0.449         | 0.232         |


Key findings:

- Text-only models remained weak across all encoders.
- Bio_ClinicalBERT fusion had the strongest accuracy among the text-encoder fusion runs.
- CheXbert fusion had the strongest macro F1 in the seed-42 comparison.
- The lack of a single winner meant the next experiment needed to test fusion architecture, not just encoder choice.

Discovery: encoder choice changes the tradeoff between accuracy and balanced class performance, but text encoder selection alone is not enough.

### Notebook 03: Residual Gated Fusion Improved The Multimodal Backbone

Notebook 03 replaced the standard fusion layer with a Bio_ClinicalBERT residual gated fusion block.


| Model                                     | Test Accuracy | Test Macro F1 | Validation Accuracy | Validation Macro F1 |
| ----------------------------------------- | ------------- | ------------- | ------------------- | ------------------- |
| Bio_ClinicalBERT residual gated fusion    | 0.775         | 0.409         | 0.778               | not recorded        |
| Bio_ClinicalBERT standard fusion baseline | 0.763         | 0.311         | 0.784               | 0.433               |
| Dual-view image baseline                  | 0.756         | 0.360         | 0.767               | 0.431               |
| Bio_ClinicalBERT text-only baseline       | 0.502         | 0.203         | 0.496               | 0.241               |


Key findings:

- Residual gated fusion produced the strongest held-out test result in this comparison.
- The macro-F1 improvement over standard Bio_ClinicalBERT fusion was the most important signal: 0.409 vs 0.311.
- The validation caveat remained: the gated model did not dominate every validation metric.

Discovery: gating helped enough to carry forward as the preferred architecture, but it was still not a fully stable final answer.

### Notebook 04: Segmentation Did Not Beat Full Images

Notebook 04 tested whether lung segmentation preprocessing would improve the residual gated fusion model.


| Image Pipeline                             | Validation Accuracy | Validation Macro F1 | Test Accuracy | Test Macro F1 |
| ------------------------------------------ | ------------------- | ------------------- | ------------- | ------------- |
| Original full frontal and lateral          | 0.780               | 0.487               | 0.737         | 0.420         |
| Lung-cropped frontal plus original lateral | 0.765               | 0.383               | 0.765         | 0.384         |
| Lung-masked frontal and lateral            | 0.303               | 0.216               | 0.286         | 0.194         |


Key findings:

- Cropped frontal images improved raw test accuracy relative to original full images, but reduced test macro F1.
- The original full-image pipeline had the best validation behavior and best held-out test macro F1.
- The masked-dual pipeline collapsed, suggesting the mask removed or distorted useful diagnostic context.
- Cropped frontal preprocessing was especially risky because pleural-effusion test F1 fell to 0.000.

Discovery: segmentation is useful as an ablation, but full images should remain the default because diagnostic context outside the segmented lung region appears to matter.

### Notebook 05: Tuning Improved Validation Balance But Not Final Test Generalization

Notebook 05 tuned the selected Bio_ClinicalBERT residual gated fusion model while keeping the original full-image pipeline.

The notebook-aligned output folder is notebooks_outputs/experiment_05_gated_hypertuning_augmentation/.

Best validation trial:


| Trial                            | Validation Macro F1 | Validation Accuracy | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| -------------------------------- | ------------------- | ------------------- | ------------- | ------------- | ---------------- |
| intensity_jitter_regularized_mix | 0.533               | 0.782               | 0.674         | 0.408         | 0.685            |


The selected tuning trial used intensity jitter, stronger weight decay, higher dropout, and softer class weighting. It improved validation macro F1 over the Notebook 04 original full-image baseline, but it did not improve held-out test performance.

Per-class test F1 for the tuned checkpoint:


| Class                | Precision | Recall | F1    |
| -------------------- | --------- | ------ | ----- |
| Cardiomegaly         | 0.435     | 0.526  | 0.476 |
| Chronic lung disease | 0.385     | 0.294  | 0.333 |
| Normal               | 0.851     | 0.809  | 0.829 |
| Pleural effusion     | 0.357     | 0.192  | 0.250 |
| Pneumonia/opacity    | 0.127     | 0.196  | 0.154 |


Discovery: tuning changed the class tradeoff and improved validation macro F1, but the test set did not confirm better generalization. The Notebook 04 full-image checkpoint remains the stronger final model choice.

### Notebook 06: Grad-CAM Turned Metrics Into Case-Level Evidence

After the segmentation ablation selected the original full-image pipeline and Notebook 05 tuning did not displace it, Notebook 06 generated frontal-branch Grad-CAM overlays for the retained Bio_ClinicalBERT residual gated fusion checkpoint. The Grad-CAM sweep reproduced the selected full-image checkpoint's held-out test profile: 0.737 accuracy and 0.420 macro F1.

Confusion matrix from the 472 held-out test cases:


| True Label           | Predicted Cardiomegaly | Predicted Chronic Lung Disease | Predicted Normal | Predicted Pleural Effusion | Predicted Pneumonia/Opacity |
| -------------------- | ---------------------- | ------------------------------ | ---------------- | -------------------------- | --------------------------- |
| Cardiomegaly         | 13                     | 0                              | 21               | 3                          | 1                           |
| Chronic lung disease | 0                      | 4                              | 10               | 0                          | 3                           |
| Normal               | 14                     | 0                              | 317              | 6                          | 8                           |
| Pleural effusion     | 4                      | 0                              | 15               | 5                          | 2                           |
| Pneumonia/opacity    | 2                      | 0                              | 33               | 2                          | 9                           |


Key findings:

- The selected checkpoint correctly classified 348 of 472 test cases.
- It missed 79 abnormal cases by predicting normal.
- Normal recall was strong: 317 of 345 normal studies were classified as normal.
- Minority-class recall remained weak: 13 of 38 cardiomegaly, 4 of 17 chronic lung disease, 5 of 26 pleural effusion, and 9 of 46 pneumonia/opacity cases were correctly classified.
- Grad-CAM activation was often absent. A total of 307 of 472 maps, or 65.0%, had zero activation after the ReLU step.
- Correct abnormal cases with nonzero Grad-CAM signal often showed much stronger CAM intensity than incorrect cases in the same true class.

Average CAM intensity by true class and correctness:


| True Label           | Correct?  | Mean CAM | Mean Confidence |
| -------------------- | --------- | -------- | --------------- |
| Cardiomegaly         | incorrect | 0.042    | 0.806           |
| Cardiomegaly         | correct   | 0.355    | 0.885           |
| Chronic lung disease | incorrect | 0.029    | 0.860           |
| Chronic lung disease | correct   | 0.370    | 0.921           |
| Normal               | incorrect | 0.270    | 0.742           |
| Normal               | correct   | 0.014    | 0.971           |
| Pleural effusion     | incorrect | 0.072    | 0.909           |
| Pleural effusion     | correct   | 0.192    | 0.902           |
| Pneumonia/opacity    | incorrect | 0.040    | 0.891           |
| Pneumonia/opacity    | correct   | 0.120    | 0.889           |


Discovery: when the model correctly identifies abnormal cases, Grad-CAM can provide plausible visual support. But many confident normal predictions have blank or weak maps, so Grad-CAM should be treated as qualitative failure analysis rather than proof of reliable localization.

# **Case-Level Findings**

Note: Grad-CAM heatmap PNG files are generated local artifacts and are not included in this GitHub copy. The summary tables and case interpretations are retained.

### Pattern 1: The Final Model Is Normal-Biased

The most important case-level discovery is that missed abnormal cases usually collapse into the normal class. Out of 124 total errors in the selected Grad-CAM test sweep, 79 were abnormal-to-normal misses:


| True Abnormal Class  | Predicted Normal Count |
| -------------------- | ---------------------- |
| Cardiomegaly         | 21                     |
| Chronic lung disease | 10                     |
| Pleural effusion     | 15                     |
| Pneumonia/opacity    | 33                     |


This explains why accuracy can remain respectable while macro F1 stays modest. The model is usually right on normal cases, but it does not detect enough minority abnormal cases to be considered reliable for balanced diagnostic classification.

### Pattern 2: Correct Abnormal Cases Are The Most Useful Grad-CAM Examples

Correct abnormal studies with nonzero Grad-CAM activation are the best examples for the report because they show where the model appears visually grounded.


| UID  | True Label           | Prediction           | Confidence | Mean CAM | Case Interpretation                                                                                                                                                                                                                   | Heatmap                                     |
| ---- | -------------------- | -------------------- | ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 3829 | cardiomegaly         | cardiomegaly         | 0.967      | 0.557    | Report notes stable moderate cardiomegaly and prominent central pulmonary vasculature. This is a strong positive example where the prediction, report text, and CAM intensity align.                                                  | Generated locally; heatmap PNG not included |
| 2169 | chronic_lung_disease | chronic_lung_disease | 0.919      | 0.397    | Report describes hyperexpanded lungs, a large right upper-lung lucency, severe emphysema, and biapical scarring. This is a useful case showing chronic lung disease can be detected when the image features are visually distinctive. | Generated locally; heatmap PNG not included |
| 25   | pleural_effusion     | pleural_effusion     | 0.974      | 0.245    | Report describes left lower-lobe airspace disease with moderate left and small right pleural effusions. This is the cleanest pleural-effusion success case among the high-CAM examples.                                               | Generated locally; heatmap PNG not included |
| 1166 | pneumonia_or_opacity | pneumonia_or_opacity | 0.937      | 0.237    | Report describes low lung volumes with streaky bibasilar opacities, likely atelectasis over infiltrate. This is a useful pneumonia/opacity success case, though the finding is subtle and mixed with atelectasis.                     | Generated locally; heatmap PNG not included |


Interpretation: these cases support the value of multimodal gated fusion. The model can use visual signal in clinically plausible abnormal studies, especially when the radiographic abnormality is explicit in the report and visible in the frontal view.

### Pattern 3: High-Confidence Misses Often Have Blank Grad-CAM Maps

Several of the most important failure cases are highly confident abnormal-to-normal misses with zero CAM activation. These are dangerous from an interpretability standpoint because the model appears certain while providing no useful frontal localization signal.


| UID  | True Label           | Prediction | Confidence | Mean CAM | Case Interpretation                                                                                                                                                                                                                              | Heatmap                                     |
| ---- | -------------------- | ---------- | ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| 1702 | cardiomegaly         | normal     | 1.000      | 0.000    | Metadata labels the study as borderline cardiomegaly, but the impression says no acute cardiopulmonary process. This case may represent a mild/borderline finding that the simplified label treats as abnormal.                                  | Generated locally; heatmap PNG not included |
| 2532 | cardiomegaly         | normal     | 1.000      | 0.000    | Report describes cardiomegaly, possible small bilateral pleural effusions, and pulmonary opacities suggestive of edema. This is a clearer miss: a multi-finding abnormal study was absorbed into normal.                                         | Generated locally; heatmap PNG not included |
| 1728 | pneumonia_or_opacity | normal     | 1.000      | 0.000    | Impression notes that after further radiologist review, there is a right upper-lobe focal opacity likely reflecting pneumonia. This is a clinically meaningful miss and a strong example of the model failing on a subtle/localized abnormality. | Generated locally; heatmap PNG not included |
| 2664 | pleural_effusion     | normal     | 1.000      | 0.000    | The label is pleural effusion, but the impression says no definite pleural effusion is seen and instead describes mild left-base streaky opacity. This is likely a label-noise or label-priority problem rather than a clean model miss.         | Generated locally; heatmap PNG not included |


Interpretation: these failures explain why the project should not rely on accuracy or confidence alone. The model can be confidently normal on abnormal-labeled studies, and the Grad-CAM output may be blank exactly when a reviewer would want an explanation.

### Pattern 4: Apparent False Positives Reveal Label Ambiguity

Some normal-labeled cases predicted as abnormal are not obviously false positives when the report text is inspected. This suggests the compact label mapping can hide clinically relevant findings.


| UID  | True Label | Prediction   | Confidence | Mean CAM | Case Interpretation                                                                                                                                                                                                                               | Heatmap                                     |
| ---- | ---------- | ------------ | ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 3819 | normal     | cardiomegaly | 0.835      | 0.462    | Although the compact label is normal, the report describes a markedly enlarged cardiac silhouette and continued severe cardiomegaly and/or pericardial effusion. The model prediction may be clinically plausible despite being scored incorrect. | Generated locally; heatmap PNG not included |
| 1760 | normal     | cardiomegaly | 0.669      | 0.531    | The report notes mild cardiomegaly and a small area of platelike atelectasis, but the impression says no active disease. This is an example where the project label may privilege "active disease" over chronic or mild abnormalities.            | Generated locally; heatmap PNG not included |
| 916  | normal     | cardiomegaly | 0.556      | 0.457    | The report says heart size is at the upper limits of normal. The model's cardiomegaly prediction may be an overcall, but the case is borderline rather than cleanly normal.                                                                       | Generated locally; heatmap PNG not included |


Interpretation: the model is not merely making arbitrary false positives. Some mistakes expose disagreement between compact labels, MeSH tags, findings, and impressions. A stronger future analysis should audit labels before treating every apparent error as model behavior.

# **What Was Actually Discovered**

**The notebooks discovered five main things:**

1. Image features are the backbone of performance. Text-only models were consistently too weak, but text can improve fusion behavior when paired with images.
2. Gated multimodal fusion is useful. The residual gated fusion model improved held-out macro F1 over the standard Bio_ClinicalBERT fusion baseline and became the model family carried forward.
3. Segmentation preprocessing is not automatically better. Lung cropping improved raw accuracy in one comparison, but full images had better balanced performance. Lung masking performed poorly, likely because it removed context or introduced artifacts.
4. The selected model's main weakness is not random error; it is missed abnormality. Most mistakes are abnormal studies predicted as normal, especially pneumonia/opacity and cardiomegaly.
5. Grad-CAM is useful but fragile. Correct abnormal cases can show meaningful localization, but 65% of test Grad-CAM maps were blank after ReLU. Blank heatmaps are common enough that Grad-CAM should support qualitative review, not serve as proof of clinical localization.



# **Final Model Judgment**

The final project recommendation should remain the original full-image Bio_ClinicalBERT residual gated fusion checkpoint selected by the segmentation ablation, interpreted through the post-segmentation Grad-CAM case analysis.

Why:

- It has the best held-out test macro F1 among the final-stage candidates: 0.420.
- It uses the original full-image pipeline, which had the strongest validation macro F1 in the segmentation ablation.
- It avoids the failed masked-dual segmentation pipeline.
- It outperforms the tuned Notebook 05 checkpoint on held-out test accuracy and macro F1.
- Its case-level behavior is understandable: strong normal performance, occasional visually grounded abnormal detections, and a clear abnormal-to-normal miss pattern.

The Notebook 05 tuned model should not be promoted as final. It improved validation macro F1 to 0.533, but held-out test accuracy dropped to 0.674 and test macro F1 dropped to 0.408. That is evidence of validation/test mismatch, not a final-model improvement.

# **Research Relevance**

This project is relevant to medical research because it tests whether multimodal learning can better reflect how radiology decisions are made. The results show that report text and gated fusion can improve model behavior, but they also show why medical AI needs more than accuracy before it can be trusted.

The case analysis identifies clinically important failure patterns, especially abnormal studies predicted as normal and ambiguous labels that affect model evaluation. These findings make the project useful as a research workflow for studying model reliability, dataset quality, and interpretability in chest X-ray classification.

# **Future Research**

Future work should focus on reliability rather than another broad model sweep. The priority is a structured review of the 124 incorrect Grad-CAM test cases, especially abnormal studies predicted as normal. That review should separate true model failures from label ambiguity, using the report text, predicted confidence, CAM status, and representative cases such as UID 2664, 3819, and 1760.

After the label audit, the most useful experiments are calibration and minority-class targeting: class-specific thresholds, normal-class suppression or abstention rules, class-aware sampling, focal loss, or reweighted cross-entropy for pneumonia/opacity, cardiomegaly, and pleural effusion. Grad-CAM should remain a qualitative review tool, with blank-CAM frequency tracked by class and prediction type, but it should not be presented as proof of clinically valid localization.

# **Conclusion**

Across the notebook sequence, the project moved from simple baselines to a recoverable multimodal model selection. The strongest supported model is Bio_ClinicalBERT residual gated fusion on original full frontal and lateral images. The model performs well enough to show that multimodal gated fusion is useful, but the case analysis shows why it is not yet robust: it still misses many abnormal studies as normal, and some high-confidence misses have blank Grad-CAM explanations.

The most honest final interpretation is that the project succeeded in identifying the best current architecture and the main failure mode. It did not produce a clinically reliable diagnostic classifier. The next scientific step is not another broad sweep, but a focused minority-class error and label-audit analysis using the case-level Grad-CAM artifacts already produced after the segmentation ablation.