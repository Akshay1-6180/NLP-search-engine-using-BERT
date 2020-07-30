CONTENT Search Engine

Brief
-----
i have given  python files for the query model and a static folder and templates folder .The static folder contains css files while the templates folder containes html files for both the content and the query.The model is taken from https://github.com/UKPLab/sentence-transformers and the model was remotely placed in a model folder and the pretrained model parameters was downloaded from 



# model 
The files for the textrank + T5 text summarizer + BERT embedding model by Akshay are:
1. data_provider_2
2. api_2
3. model_summarizer

Setup Instructions for content query
-------------------------------------------------
    1. First installed pip3 and pip sudo apt-get install python3-pip
    2. then my first action was to make python version to 3.6 since it was necessary to run pytorch and sentence_transformers library hence I decided to install python3.6 .But python 3.6 wasnt readily available to ubuntu 16.04 hence i instaled it by a indiect method
    3. sudo apt-get install software-properties-common
    4. sudo add-apt-repository ppa:deadsnakes/ppa 
    5. sudo apt-get update
    6. sudo apt-get install python3.6
    7. installed a pakcage called virtualenv  using sudo pip3 install virtualenv
    8.  To make a enviornment to install all packages there since i needed a lot of packages to install and hence i didnt want to make any changes to the sytem libraries and if anything wrong happens using installation you can delete the folder and repeat
    9. I created the enviornment by typing virtualenv --python=/usr/bin/python3.6 venv
    10. created a enviornment  and got into its enviornment do source venv/bin/activate
    11. to go out of the enviornment after ur work is done type deactivate (do not type it now only if u want to go to the system libraries)
    12. then  pip3 --no-cache-dir install torchvision to install torchvision library which would also install the libraries future,numpy,pillow,torch
    13. pip3 install sentence_transformers which installed the following packages threadpoolctl, scipy, scikit-learn, nltk, sentence-transformers
    14. pip3 install mysql-connector-python-rf
    15. pip3 install pandas
    16. pip3 install -U textblob
    17. pip3 install Flask
    18. pip3 install BeautifulSoup4
    19. pip3 install transformers
    20.pip3 install --upgrade gensim

# for the content query 

Build Question Model
-------------
#please make the changes in the sql connect before running the files to connect to ur database

#for model 

Run the data_provider2 file which would make the final_content.csv file which makes two new columns of word count and sentence that are text ranked  by connecting to the sql 
mysql.connector.connect(host='localhost',
						database={enter the name of the questions database},
                        user={enter the user }
                        password = {enter the password})

Build Content Model
-------------
#this is only for model
after running data_provider_2.py run model_and_summarizer.py which would create summary_content.csv which has the summary for all the text and then it also creates a embedding for the article summary and saves it in a file called sentence_encoder_content.Currently it gives a summary for all the articles and it takes 3 hours in GPU so if any new articles are added please do the summary only for those and add it and dont run it again.
run the model embedding after that.
then it created another file called summary_content.csv

Running API Server
---------------
#for model_1
run the api_1.py and set the server accordingly to local host or any other server.running it will take around 30 seconds and after that displaying the results will take around 3-5 seconds.
make sure the following files are made before running
1. dict_file_sentence_wise_with_embedding
2. summary_content.csv
#for model_2
run the api_2.py and set the server accordingly to local host or any other server.running it will take around 30 seconds and after that displaying the results will take around 3-5 seconds.
1. summary_content.csv
2. sentence_encoder_content


                                                  database='parentlane',
                                                  user='root'
                                                  password= '*****'')
