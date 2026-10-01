# Data attribution and demo scope

The dashboard displays selected research examples, not an unbiased performance estimate, a diagnosis, or a validated clinical assessment.

- IXI Dataset, Imperial College London: https://brain-development.org/ixi-dataset/ . Dataset license: CC BY-SA 3.0. The original MRI hashes match the manifests used in this project. Dataset examples IXI170, IXI361 and IXI566 were selected after held-out evaluation.
- Dallas Lifespan Brain Study (DLBS), OpenNeuro ds004856 v1.2.0: https://openneuro.org/datasets/ds004856/versions/1.2.0 . Dataset license recorded in the catalog: CC0 1.0. Selected external example: sub-3898.
- OASIS-2: https://sites.wustl.edu/oasisbrains/home/oasis-2/ . Follow the dataset's usage terms and citation requirements; public availability does not imply unrestricted use. OAS2_0007's actual recorded visits are MR1 and MR3, 518 days apart.

Signed Grad-CAM is coarse post-hoc model attribution, not an anatomical segmentation or a disease-risk map. Regional volumes are FastSurfer bilateral DKT anatomical gray-matter groups. Cortical groups exclude white matter, cingulate and insula. Technical visual inspection is not clinical validation.

The website is a precomputed static demonstration; it does not upload or analyze a visitor's MRI.
