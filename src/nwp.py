# @author: Dilsad Ergun Kucukkececi

import random #random library for text generation process
import re #re library for pre process of sets and post process of generated text
import math


"""
    Method is for pre processing phase of the train data sets
    Each author's training set is described
    Regular expressions are mainly used on
"""
def preProcessAuthor(authorName):
    
    s="" #train data is defined empty at the beginning

    if authorName=="Hamilton":

        hamiltonTrainingSet=[6, 7, 8, 13, 15, 16, 17, 21, 22, 23, 24, 25, 26 ,27, 28,29] #train files for Hamilton
        
        s = open("data/1.txt", 'r').read() #open first file and read into train data
        s=s.split('\n')[1] #split according to new line to avoid author name
        for i in hamiltonTrainingSet: #read other files in a loop
            temp=open("data/"+str(i)+".txt", 'r').read()
            temp=temp.split('\n')[1]
            s+="  "+ temp; #append each data
            temp=""


    elif authorName=="Madison":
        
        madisonTrainingSet=[14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46] #train files for Madison
        
        s = open("data/10.txt", 'r').read() #open first file and read into train data
        s=s.split('\n')[1]
        for i in madisonTrainingSet: #read other files in a loop
            temp=open("data/"+str(i)+".txt", 'r').read()
            temp=temp.split('\n')[1]
            s+="  "+ temp; #append each data
            temp=""
    else:
        print("Undefined author")
        return

        
    s = re.sub('[()]', r'', s)            # remove certain punctuation chars
    s = re.sub('([.-])+', r'\1', s)       # collapse multiples of certain chars

    """
        Define end of sentence with </s> symbol
    """

    s=s.replace('.', ' </s> ')
    s=s.replace('?', ' </s> ')
    s=s.replace('!', ' </s> ')



    s = ' '.join(s.split()).lower() # remove extra whitespace (incl. newlines) and turn characters to lower

    """
        Different punctuation char replacements
    """
        
    s=s.replace(',', ' ')
    s=s.replace(':', ' ')
    s=s.replace(';', ' ')
    s=s.replace('``', ' `` ')
    s=s.replace('-', ' ')
    s=s.replace('\'', ' ')
    s=s.replace('[', ' ')
    s=s.replace(']', ' ')
    s=s.replace('@' , '@ ')
    s=s.replace('%', ' %')


    s=re.sub(' +', ' ', s) #remove all extra spaces


    return s

    """
        Method for creating n-gram twins
    """

def createNGrams(trainSet,N):
    
    tokens=trainSet.split(" ") #split by blank, get all words
    ngrams = []                #array to hold all ngrams without counts

    
    for i in range(len(tokens)-N+1): #walk in tokens related to N value
        ngrams.append(tokens[i:i+N]) #append tokens with N-1 groups
    return ngrams


    """
        Method for calculating n-gram counts
        Creates nested dictionary structure for each N-gram
        Gets the n-gram array as parameter
    """

def createNGramModel(ngrams):

    ngramDictionary = {}    #initialize dictionary
    
    for ngram in ngrams:    #iterate all n-grams
        sequence  = " ".join(ngram[:-1])    #choose leader token
        end = ngram[-1]                     #choose follower token
        
        
        if sequence not in ngramDictionary: #if leader not added to dictionary before
            ngramDictionary[sequence] = {}; #assign empty dictionary as value
        
        if end not in ngramDictionary[sequence]:   #check if follower exists in leader's values
            ngramDictionary[sequence][end] = 0;    #add and initliaze count
        
        ngramDictionary[sequence][end] += 1;    #increase found n-gram count
        
    return ngramDictionary

    """
    Method for finding next word in essay generation process
    Takes generated essay until now as parameter
    Takes N value for calculating opportunities
    Take related dictionary as parameter to find next
    """

def findNextWord(N, sentence, ngramCounts):
    

    if N==1: #if unigram
        opportunities=ngramCounts[""].items()  #all keys can be chosen
    else: #if bigram or trigram
        
        #only choose the probably followers
        
        sequenceOfTokens = " ".join(sentence.split()[-(N-1):]) #find last N-1 words of sentence
        opportunities=ngramCounts[sequenceOfTokens].items() #opportunities are its inner dictionary
        #probable words are the words that can come after that
    
    
    
    total = sum(weight for choice, weight in opportunities) #create probability distribution

    r = random.uniform(0, total)    #choose random


    #apply weighted random choice
    upto = 0
    for choice, weight in opportunities:
        

        upto += weight;
        if upto > r:
            
            return choice



    return choice

    """
    Method for styling of generated essay
    """

def postProcessOutputText(outputSentence):
    
    outputSentence = outputSentence.capitalize() # capitalize first letter
    if(outputSentence.endswith('</s>')): #if there is end of sentence symbol
       outputSentence=outputSentence.replace(' </s>', '.') #end the sentence with punctuation
    
    else: #if no symbol means the sentence is finished with 30 words limit
        outputSentence+='.' #end the sentence with punctuation

    outputSentence=re.sub(' +', ' ', outputSentence) #remove extra spaces
    
    return outputSentence


    """
    Method for generating whole essay
    N defines the n-gram model to use
    relatedDictionary is the ngram counts of one author
    """

def generateEssay(N, relatedDictionary):
    
    
    generatedEssay=""
    numOfWords=0 #to keep track of 30 word limit
    
    #choose a random start from keys
    
    if N==1:
        randomStart=random.choice(list(relatedDictionary[""].keys()))
    else:
    
        randomStart=random.choice(list(relatedDictionary.keys()))
    
    
    
    generatedEssay+=randomStart
    
    while numOfWords<30:    #until reaching word limit
        
        if generatedEssay.endswith(('.','!', '?', '</s>')): #if sentence is ended
            return generatedEssay
        
        
        generatedEssay += " " + findNextWord(N, generatedEssay, relatedDictionary) #append next word to essay
        numOfWords+=1
        
        

    return generatedEssay

    """
    Method for calculating probability for each n-gram pair
    Applies laplace smoothing before returning each probability
    V is specific to n-gram model and author
    """

def calculateProbability(relatedDictionary, wordLeader, wordFollower, V):
    
    if wordLeader in relatedDictionary.keys():  #check if the n-gram is already in data
        
        #calculata denominator
        totalCount=0;
        for i in relatedDictionary[wordLeader].values():
            totalCount+=i
        
    
        if wordFollower in relatedDictionary[wordLeader].keys(): #check if the n-gram follower is already in data
            
            #if already in train data
            
            count=relatedDictionary[wordLeader][wordFollower] #calculate share directly
        

        else:
            #if unseen pair
            count = 0
    
    else:
        #if totally unseen word
        count=0
        totalCount=0
    

    probability=(count +1) / (totalCount + V)   #apply smoothing

    probability=math.log(probability, 2)   #get log of probability to avoid underflow

    return float(probability)


    """
    Apply same pre-processing to test data
    All steps are identical
    """

def preProcessTestData(testData):

    s = open(testData, 'r').read() #read test data
    s=s.split('\n')[1] #do not take author name
    s+= "  "
    
    s = re.sub('[()]', r'', s)            # remove certain punctuation chars
    s = re.sub('([.-])+', r'\1', s)       # collapse multiples of certain chars
    
    """
        Define end of sentence with </s> symbol
    """
    
    s=s.replace('.', ' </s> ')
    s=s.replace('?', ' </s> ')
    s=s.replace('!', ' </s> ')
    
    
    
    s = ' '.join(s.split()).lower() # remove extra whitespace (incl. newlines) and turn characters to lower
    
    """
        Different punctuation char replacements
    """
    
    s=s.replace(',', ' ')
    s=s.replace(':', ' ')
    s=s.replace(';', ' ')
    s=s.replace('``', ' `` ')
    s=s.replace('-', ' ')
    s=s.replace('\'', ' ')
    s=s.replace('[', ' ')
    s=s.replace(']', ' ')
    s=s.replace('@' , '@ ')
    s=s.replace('%', ' %')
    
    
    s=re.sub(' +', ' ', s) #remove all extra spaces
    

    return s


"""
    Calculate perplexity for evaluation
"""

def calculatePerplexity(testData, relatedDictionary, relatedV, N):
    
    

    processedTestData=preProcessTestData(testData) #pre process test data to evaluate
    tokensOfTestData= processedTestData.split(" ") #get the tokens of test data
    
    probability=0
    
    i=0

    numOfMultipication=0 #shows the N is perplexity formula
    
    if N==2:    #for bigram
    
        while i<len(tokensOfTestData)-1:
            
            #probabilities are added since they return in log form

            probability+=calculateProbability(relatedDictionary, tokensOfTestData[i], tokensOfTestData[i+1], relatedV)
            #calculate probability for each leader word-follower word pair
        

            numOfMultipication+=1
            i+=1

    elif N==3:
        while i<len(tokensOfTestData)-2:
            
            #probabilities are added since they return in log form
            
            probability+=calculateProbability(relatedDictionary, tokensOfTestData[i] + " " + tokensOfTestData[i+1], tokensOfTestData[i+2], relatedV)
        
            #calculate probability for each leader word sequence-follower word pair
            
            numOfMultipication+=1
            i+=1

    perplexity=(2**((-1/numOfMultipication)*(probability))) #formula = 2^((-1/N)*log2P(total))

    return perplexity


    """
    Method for comparing generated essays by an author with the author's n-gram models
    Probabilities are calculated to compare n-gram's succeess in different essays
    """
def evaluateGeneratedEssay(essay, relatedDictionary, N, V):
    
    tokensOfTestData= essay.split(" ") #tokenize essay
    
    probability=0
    totalCount=0
    count=0
    i=0
    
    if N==1: #for unigram
        
        #since essay is generated from this dictionary all words exist
        
        for i in relatedDictionary[""].values():
            totalCount+=i
    
        while i<len(tokensOfTestData):
            count=relatedDictionary[""][tokensOfTestData[i]]
            i+=1
            probability+=math.log((count+1)/(totalCount+V), 2) #apply smoothing and log operations, add all

    elif N==2: #for bigram
        
        #different combinations can exist so checks need to be done
        
        while i<len(tokensOfTestData)-1:
            
            probability+=calculateProbability(relatedDictionary, tokensOfTestData[i], tokensOfTestData[i+1], V)
            #use same probability method defined above
            i+=1

    elif N==3: #for trigram
        
        #different combinations can exist so checks need to be done
        
        while i<len(tokensOfTestData)-2:
            probability+=calculateProbability(relatedDictionary, tokensOfTestData[i] + " " + tokensOfTestData[i+1], tokensOfTestData[i+2], V)
            #use same probability method defined above
            i+=1

    else: print("Non defined n gram")

    return(probability)


"""
    Method for authorship detection
    Perplexities are compared to classify test inputs
"""

def classifyText(testText, N , hamiltonNgramCounts, madisonNgramCounts, V_Hamilton, V_Madison):

    author=""
    
    #calculate perplexities with hamilton model
    hamiltonBiResult=calculatePerplexity(testText, hamiltonNgramCounts, V_Hamilton, N)
    #display perplexity
    print("Perplexity for Author Hamilton: " + str(hamiltonBiResult))
    
    #calculate perplexities with madison model
    madisonBiResult=calculatePerplexity(testText, madisonNgramCounts, V_Madison, N)
    #display perplexity
    print("Perplexity for Author Madison: " + str(madisonBiResult))
    
    #calculations to revert percentages
    totalPerplexity= hamiltonBiResult + madisonBiResult
    multiplier = 100/totalPerplexity
    
    madisonPercentage= madisonBiResult*multiplier
    hamiltonPercentage= hamiltonBiResult*multiplier
    print("")
    

    
    """
        Revert perplexities to percentages with inverse ratio
        Smaller perplexity=higher percentage
    """
    
    #display results
    print("%"+ str(100-hamiltonPercentage) + " author is Hamilton")
    print("")

    print("%"+ str(100-madisonPercentage) + " author is Madison")

    #return results
    if hamiltonBiResult<madisonBiResult:
        author="Hamilton"
    else: author="Madison"

    return author


if __name__ == "__main__":  # main program
    
    print("-------TASK 1-------")
    print("")
    
    #pre process
    hamiltonSet=preProcessAuthor("Hamilton");
    madisonSet=preProcessAuthor("Madison");
    #pre process

    #create unigrams
    hamiltonUnigrams=createNGrams(hamiltonSet, 1)
    hamiltonUniCounts=createNGramModel(hamiltonUnigrams)
    
    madisonUnigrams=createNGrams(madisonSet, 1)
    madisonUniCounts=createNGramModel(madisonUnigrams)
    #create unigrams
    
    
    #calculate distinct word count for smoothing
    V_Hamilton_UniBiGram = 0
    V_Madison_UniBiGram=0
    V_Hamilton_TriGram=0
    V_Madison_TriGram=0
    for i in hamiltonUniCounts[""].keys():
        V_Hamilton_UniBiGram+=1
    for i in madisonUniCounts[""].keys():
        V_Madison_UniBiGram+=1
    #V is used for unigram-bigram probabilities


    
    #create bigrams
    hamiltonBigrams=createNGrams(hamiltonSet, 2)
    hamiltonBiCounts=createNGramModel(hamiltonBigrams)

    madisonBigrams=createNGrams(madisonSet, 2)
    madisonBiCounts=createNGramModel(madisonBigrams)
    #create bigrams
    

    #create trigrams
    hamiltonTrigrams=createNGrams(hamiltonSet, 3)
    hamiltonTriCounts=createNGramModel(hamiltonTrigrams)
    
    madisonTrigrams=createNGrams(madisonSet,3)
    madisonTriCounts=createNGramModel(madisonTrigrams)
    #create trigrams

    #calculate distinct bigram count for smoothing
    for i in hamiltonBiCounts.keys():
        for j in hamiltonBiCounts[i]:
            V_Hamilton_TriGram+=1

    for i in madisonBiCounts.keys():
        for j in madisonBiCounts[i]:
            V_Madison_TriGram+=1
     #V is used for trigram probabilities



    print("N-grams are created")
    print("")

    print("-------TASK 2-------")
    print("")

    print("Generated essay-1 with unigrams for author Hamilton:")
    print("")

    essay=generateEssay(1, hamiltonUniCounts) #generate essay with unigrams
    print(postProcessOutputText(essay)) #post-process
    print("")




    print("Generated essay-2 with unigrams for author Hamilton:")
    print("")
    essay=generateEssay(1, hamiltonUniCounts) #generate essay with unigrams
    print(postProcessOutputText(essay)) #post-process
    print("")

    print("Generated essay-1 with bigrams for author Hamilton:")
    print("")
    essay=generateEssay(2, hamiltonBiCounts) #generate essay with bigrams
    print(postProcessOutputText(essay)) #post-process
    print("")

    print("Generated essay-2 with bigrams for author Hamilton:")
    print("")
    essay=generateEssay(2, hamiltonBiCounts) #generate essay with bigrams
    print(postProcessOutputText(essay)) #post-process
    print("")

    print("Generated essay-1 with trigrams for author Hamilton:")
    essay=generateEssay(3, hamiltonTriCounts) #generate essay with trigrams
    print(postProcessOutputText(essay)) #post-process
    print("")

    print("Generated essay-2 with trigrams for author Hamilton:")
    essay=generateEssay(3, hamiltonTriCounts) #generate essay with trigrams
    print(postProcessOutputText(essay)) #post-process
    print("")

    print("Generated essay-1 with unigrasms for author Madison:")
    print("")
    essay=generateEssay(1, madisonUniCounts) #generate essay with unigrams
    print(postProcessOutputText(essay)) #post-process

    print("")
    print("Generated essay-2 with unigrasms for author Madison:")
    print("")
    essay=generateEssay(1, madisonUniCounts) #generate essay with unigrams
    print(postProcessOutputText(essay)) #post-process
    print("")

    print("Generated essay-1 with bigrams for author Madison:")
    print("")
    essay=generateEssay(2, madisonBiCounts) #generate essay with bigrams
    print(postProcessOutputText(essay)) #post-process

    print("")
    print("Generated essay-2 with bigrams for author Madison:")
    print("")
    essay=generateEssay(2, madisonBiCounts) #generate essay with bigrams
    print(postProcessOutputText(essay)) #post-process

    print("Generated essay-1 with trigrams for author Madison:")
    print("")
    essay=generateEssay(3, madisonTriCounts) #generate essay with trigrams

    print(postProcessOutputText(essay)) #post-process
    print("")

    print("Generated essay-2 with trigrams for author Madison:")
    print("")
    essay=generateEssay(3, madisonTriCounts) #generate essay with trigrams
    
    print(postProcessOutputText(essay)) #post-process

    #evaluate last generated essay as example
    #probablities is in log form
    #calculated for comparisons

    print("")
    print("Example essay evaluation for Madison's 3-gram essay")
    print("")
    print("Unigram probability for essay "+str(evaluateGeneratedEssay(essay, madisonUniCounts, 1, V_Madison_UniBiGram)))
    print("Bigram probability for essay "+ str(evaluateGeneratedEssay(essay, madisonBiCounts, 2, V_Madison_UniBiGram)))
    print("Trigram probability for essay "+ str(evaluateGeneratedEssay(essay, madisonTriCounts, 3, V_Madison_TriGram)))

    print("")
    print("-------TASK 3-------")
    print("")

    #Classify test inputs with 2-gram models
    print("Classification with 2-grams")
    print("")
    testInputArray=["data/9.txt", "data/11.txt","data/12.txt", "data/47.txt", "data/48.txt", "data/58.txt"]

    for testInput in testInputArray:
        print("Current test input file: " + testInput)
        print("")
        classifyText(testInput, 2 , hamiltonBiCounts, madisonBiCounts, V_Hamilton_UniBiGram, V_Madison_UniBiGram)
        print("")
    print("")

    #Classify test inputs with 3-gram models
    print("Classification with 3-grams")
    for testInput in testInputArray:
        print("Current test input file: " + testInput)
        print("")
        classifyText(testInput, 3 , hamiltonTriCounts, madisonTriCounts, V_Hamilton_TriGram, V_Madison_TriGram)
        print("")
    print("")

    print("Example of unknown file classification")
    testInput="data/62.txt"

    print("")
    
    #Classify unknown file with 2-gram models
    print("Classification with 2-grams")
    print("")
    classifyText(testInput, 2 , hamiltonBiCounts, madisonBiCounts, V_Hamilton_UniBiGram, V_Madison_UniBiGram)
    print("")
    #Classify unknown file with 3-gram models
    print("Classification with 3-grams")
    print("")
    classifyText(testInput, 3 , hamiltonTriCounts, madisonTriCounts, V_Hamilton_TriGram, V_Madison_TriGram)






