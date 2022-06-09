## AutoML framework benchmark - AutoKeras 
[AutoKeras website](https://autokeras.com/), [AutoKeras article](https://arxiv.org/pdf/1806.10282.pdf)


### Directory architecture
There are 4 directories:

* article 
* aresentation
* code
* results

In the Article directory there is a .pdf which is our final article and in the directory 'Pictures' there are pictures used in the final article.

In the Presentation directory there is only a file .pdf which is our final presentation.

In the code directory there are four files:

* `exec.py` - majority of functions used in model search and training
* `wyniki.ipynb` - loads and presents results 
* `GenerateModelArchitecture.ipynb` - generates model architecture diagram
* `customised_model.ipynb` - AutoModel based on AutoKeras API


In the results directory there are four directories which names describe the settings of AutoKeras models used in benchmarking. In each of them there is a 'results.json' file in which there are metrics score for each dataset, as well as elapsed time.  

### How to run the scripts
To run the benchmark one simply needs to execute `exec.py` file. We used %run command in `wyniki.ipynb` file.

To view the results in `wyniki.ipynb` you should add the `results` folder to the path of each `results.json`, and execute the file.

To run model architecture visualizer, use `GenerateModelArchitecture.ipnyb` and change last few cells to your needs.





