# Tweetopedia_Collab_Notebooks

This is a collection of all the notebooks that I used in order to train Neural Network Models for my web Project TweetoPedia.If you want to play around and contribute to the project by optimizing the neural network parameters go and follow the instructions given below.The Dependencies for this project are already given in the requirements.txt file in the Django Project.If you havent installed follow my instructions in the original Project to install it.


## Query based Sentiment Analysis
Open <a href = "https://github.com/avinsit123/Tweetopedia_Collab_Notebooks/blob/master/Query_based_Convolutional_Sentiment_Analysis.ipynb" >this notebook</a> in collab by clicking the button on top of the notebook.
I have used a Convolutional Neural Network to perform sentiment Analysis.Go Ahead and Read the Notebook to understand more about how the process has been carried out.Run all the cells in a sequential order until you reach the cell with the contents
```python
BATCH_SIZE = 64
N_EPOCHS = 5
INPUT_DIM = len(TEXT.vocab)     #Do not change this
EMBEDDING_DIM = 100             #Embedding Dims for Glove vectors 
N_FILTERS = 100
FILTER_SIZES = [3,4,5]          #Size of Convolutional Filter
OUTPUT_DIM = 1                  #Output Dimensions = 1  
DROPOUT = 0.5
device = [ "cuda" if torch.cuda.is_available() else "cpu"]
model_name = "hateorlove"       #Name of check point file
```

These are the parameters for the Neural Network.Which you can change to improve Accuracy.Finally After you run all your cells.A checkpoint file will be generated.To Download the file,open the files tab on the right portion of the window.Double-Click on the .pth file and It would automatically start downloading the file.<br>

Copy the source path of the checkpoint file on your local computer.And open the python file with the below give directory.
```
/Tweet-o-pedia/mysite/QuerybasedSentiment/views.py
```
and move to the line
```python
def goloader(tweet):
   INPUT_DIM = 25002
   EMBEDDING_DIM = 100
   N_FILTERS = 100
   FILTER_SIZES = [3,4,5]
   OUTPUT_DIM = 1
   DROPOUT = 0.5 

   model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
   with open('/Users/r17935avinash/Downloads/hateorlove.pth', 'rb') as f:
       checkpoint = torch.load(f,map_location='cpu')
```
Replace the parameters that you changed in the above line of code and replace "/Users/r17935avinash/Downloads/hateorlove.pth" with your copied source path and run the project.
