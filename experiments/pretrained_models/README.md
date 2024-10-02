# Pretrained models

Pretrained models include:

| Shortname | Description |
| --- | --- |
| `posetext_model_bedlamscript` | [PoseText](https://github.com/naver/posescript/tree/main/src/text2pose/retrieval) model trained on BEDLAM-Script | 
| `posevae_model_bedlamscript` | PoseVAE model (see *./src/poseembroider/assisting_models/poseVAE*) trained on the poses of BEDLAM-Script |
| `pairtext_model_ft_bedlamfix_posefix` | [PairText](https://github.com/naver/posescript/tree/main/src/text2pose/retrieval_modifier) model trained on BEDLAM-Fix|
| `poseembroider_model_bedlamscript` | **PoseEmbroider** model, trained on BEDLAM-Script |
| `aligner_model_bedlamscript` | **Aligner** model, trained on BEDLAM-Script |
| `instructgen_model_poseembroider_bedlamfix_posefix` | Neural head, trained on **PoseEmbroider** features derived from the 3D poses of BEDLAM-Fix & PoseFix, producing human pose corrective instructions in natural language. Model pretrained on BEDLAM-Fix, and finetuned on both datasets. |
| `instructgen_model_aligner_bedlamfix_posefix` | Neural head, trained on **Aligner** features derived from the 3D poses of BEDLAM-Fix & PoseFix, producing human pose corrective instructions in natural language. Model pretrained on BEDLAM-Fix, and finetuned on both datasets. |
| `hpsestimator_model_poseembroider_bedlamscript` | Neural head, trained on **PoseEmbroider** features derived from the images of BEDLAM-Script, estimating the 3D human pose and shape (SMPL-X parameters). |
| `hpsestimator_model_aligner_bedlamscript` | Neural head, trained on **Aligner** features derived from the images of BEDLAM-Script, estimating the 3D human pose and shape (SMPL-X parameters). |

# Setup released models

* Download all pretrained models at once (total < 1 Go) by running the following script:
    ```bash
    cd "<GENERAL_EXP_OUTPUT_DIR>" # TODO replace with your favorite location!

    # you can comment below the models you are not interested in
    arr=(
        posetext_model_bedlamscript
        posevae_model_bedlamscript
        pairtext_model_ft_bedlamfix_posefix
        poseembroider_model_bedlamscript
        aligner_model_bedlamscript
        instructgen_model_poseembroider_bedlamfix_posefix
        instructgen_model_aligner_bedlamfix_posefix
        hpsestimator_model_poseembroider_bedlamscript
        hpsestimator_model_aligner_bedlamscript
    )

    for a in "${arr[@]}"; do
        echo "Download and extract $a"
        wget "https://download.europe.naverlabs.com/ComputerVision/PoseEmbroider/${a}.zip"
        unzip "${a}.zip"
        rm "${a}.zip"
    done
    ```

* Fix the paths in *src/poseembroider/shortname_2_model_path.json* from relative paths to **absolute** ones.