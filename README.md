# tfidf-tool
This is an implementation of Python.The tool provides a simple and fast method to calculate tf-idf value.
## why use this tool?
+ the tool calculates idf value by multi processes,which is faster n times than traditional method 
+ it can calculate n-gram tf-idf value
+ and extract key words from documents

# quick start
All the input we use is in the 'input' directory.We will use 'wiki_head_10.txt' which contains 10 documents of wiki to train our model,and use 'wiki_test.txt' to test.
### get idf value
```
    doc = Document('../input/wiki_head_10.txt')
    tfidf = TFIDF(
        documents=doc,
        ngram=2,
        stop_words_path='../input/stop_words.txt',
        idf_path='../output/idf.txt'
    )
    //use 2 process and every process handle 5 docs
    tfidf.multi_pro_idf(process_num=2, p_doc_num=5)
```

Here we calculate bigram idf value from the 10 wiki docs.

TFIDF's parameter
+ **documents**:a class of Document. The input is a generator which every element is a list of sentence which represents a document
+ **ngram**:Integer.1 represents unigram, 2 represents bigram, 3 represents trigram...
+ **strop_words_path**:stop words file.If use stop words, the ngram words contain stop words will filtered.
+ **idf_path**:a file path to store the idf value
