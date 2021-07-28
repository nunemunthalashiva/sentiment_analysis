# Predicting whether a review is positive or negative.
Our goal here is to predict whether the review written by a staranger is positive or negative . We have taken the dataset from <a href = "https://www.rottentomatoes.com/"> Rotten Tomatoes</a> (polarity.train and polarity.dev) . We can approach this problem by many ways but the simplest way (which is not that effective but still..) is to just do binary classification . i.e. we are segregating reviews either positive(+1) or negative(-1) class . <br> <br>
Lets understand the step of preprocessing words to numbers . One of the straight forward way is to use all the words in the world and count the number of occurences of each word . Then we get a highly sparse vector (i.e. with many zeros) So its better to use dictionary(hash table) in python which are indxed by words . <br><br>
### Understanding workflow .

#### Counting occurences of each word in a string.
This function takes a string and returns a dictionary with the number of occurences of each word . While when we are doing dot product later we assume if there is no given key we think that the key's value is zero which is easier .

#### Hypothesis class
In general every machine learning problem starts with assuming assuming a hypothesis class . The most used function in general will be a sigmoid function (we can make this to predict between [-1,1] by taking 2*sigmoid - 1 )<br>
It turns out to be in this example we can simply use **sign(Weights*Φ(x))** to predict whether its +ve or -ve <br>
Here Φ(x) is a dictionary with keys as words and values as number of counts
<br>
#### Learning Weights (optimisation)
We have to describe our loss function before we are doing optimisation . It turns out be that its so straight forward one can think we can use Hinge loss i.e **Losshinge(x,y,w) = max{1 −(w·φ(x))*y , 0}** note here our function turns out be convex . So we can use either **gradient descent  or stochastic gradient descent **  Lets use SGD here .

#### Stochastic gradient descent
Lets first calculate gradient **∇Losshinge(x,y,w) = {−φ(x)y if 1 > {(w·φ(x))y} else 0}** So our SGD update looks like<br><br>
        **for i in range(Epochs):** <br>
             **for x,y in trainset:**<br>
                 w:= w - α * ∇Losshinge(x,y,w) <br><br>
                 
If we try to increase Epochs to large our Training error goes to zero but then our test error oscillates between 25-27 % 


