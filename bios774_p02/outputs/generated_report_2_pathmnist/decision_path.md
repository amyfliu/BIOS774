| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Release synthetic labeled images; generate | Real patches cannot be shared, so a synthetic stand-in is needed |
| 2 | Data prep | Scale pixels to [-1,1] | Standard input range for image diffusion with convolutions |
| 3 | Model | Conditional diffusion model | Best image quality; plenty of data; labels needed |
| 4 | Settings | Medium network, full data, 34 minutes | Speed test gave about 11k training steps in budget |
| 5 | Training check | Training healthy; samples look tissue-like | Loss flattened and the sample grid resembles real tissue |
| 6 | Generate | 1000 images per tissue type, mild guidance | Balanced labeled set covering all nine tissue types |
| 7 | Evaluate | Evaluated realism, copying, classifier utility | Checks whether shareable data looks real, avoids copies, trains classifiers |
| 8 | Adjust | Try stronger class guidance | Label match is only 75%; stronger guidance may raise it |
| 9 | Evaluate | Stronger guidance: clearer labels, worse realism | Label match rose but diversity and classifier use fell |
| 10 | Recommend | Use for prototyping only, with restrictions | Close to real for training, but visibly imperfect and unproven privacy |
