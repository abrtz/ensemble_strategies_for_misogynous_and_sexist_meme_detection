Once the files were downloaded, they were combined into one file to get the image captions at once with BLIP-2.

This was done in the file `preprocessing.ipynb`.

The image captions were generated with the BLIP-2 (Li et al., 2023) vision-language model in the notebook `image_preprocessing.ipynb`, which was run on Google Colab.

The resulting files were further cleaned in the preprocessing notebook above. The remaining memes that needed image-caption to be re-done were reprocessed and the final files were split into training, development and test in `preprocessing.ipynb`. The gold labels and meme paths were also added to the splits.

The final distribution of each file was further explored in the `dea.ipynb` notebook, and the files with gold labels were created to compute the respective metrics with the PyEvall library and the script from the MAMI shared task mentioned in the `/models` README.md.
