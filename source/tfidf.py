"""
@version: ??
@author: lihu.clh
@file: tfidf.py
@time: 2018/5/18 17:48
@desc:
"""
import math
import os
import sys
import nltk
import multiprocessing as mp

class Document():
    def __init__(self, doc_path):
        self.doc_path = doc_path
        if not os.path.exists(doc_path):
            print('{a} is not exist'.format(a=doc_path))
            sys.exit()
    def __iter__(self):
        with open(self.doc_path, encoding='utf8', errors='ignore') as f:
            line = f.readline()
            while line:
                yield line
                line = f.readline()

class TFIDF():

    def __init__(
            self,
            documents,
            ngram,
            idf_path,
            stop_words_path=None

    ):
        self.documnents = documents
        self.ngram = ngram
        self.ngram_dict = dict()
        self.idf_dict = {}
        self.idf_path = idf_path
        self.stop_words_path = stop_words_path
        self.ignored = {'', ' ', '', '.', ':', ',', ')', '(', '!', '?', '"'}
        if self.stop_words_path:
            self.load_stop_words()

    def load_stop_words(self):
        with open(self.stop_words_path, encoding='utf8') as f:
            line = f.readline()
            while line:
                self.ignored.add(line.strip())
                line = f.readline()

    def calculate_idf(self, args):
        out_path, documents = args
        doc_num = 0
        #count ngram
        ngram_cnt = {}
        for documnet in documents:
            doc_num += 1
            ngram_set = set()
            sentences = nltk.sent_tokenize(documnet)
            sentences = [nltk.word_tokenize(s) for s in sentences]
            for i in range(len(sentences)):
                sentence = sentences[i]
                for j in range(len(sentence)-self.ngram + 1):
                    for k in range(self.ngram):
                        if sentence[j+k] in self.ignored:
                            continue
                    ngram_word = ' '.join(sentence[j:j+self.ngram])
                    ngram_set.add(ngram_word)
            for s in ngram_set:
                if s not in ngram_cnt:
                    ngram_cnt[s] = 0
                ngram_cnt[s] += 1
        #write to file
        out_content = ''
        for k, v in ngram_cnt.items():
            out_content += k + ',' + str(v) + '\n'

        with open(out_path, 'w', encoding='utf8') as f:
            f.write(out_content)

    def multi_pro_idf(self, process_num, p_doc_num):
        temp_path = '../output/temp/'
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        pool = mp.Pool(processes=process_num)
        temp_doc_list = []
        doc_cnt = 0
        file_cnt = 1
        for doc in self.documnents:
            doc_cnt += 1
            temp_doc_list.append(doc)
            if doc_cnt % p_doc_num == 0:
                args = (temp_path+'temp_idf_' + str(file_cnt) + '.txt', temp_doc_list)
                self.calculate_idf(args)
                print('process id = {a}, start!The {b} documents'.format(a=os.getpid(),b=doc_cnt))
                pool.apply_async(func=self.calculate_idf, args=(args, ))
                temp_doc_list = []
                file_cnt += 1
        if len(temp_doc_list) > 0:
            args = (temp_path + 'temp_idf_' + str(file_cnt) + '.txt', temp_doc_list)
            pool.apply_async(func=self.calculate_idf, args=(args,))
        pool.close()
        pool.join()
        print('multi-process have done,start merge temp files')

        for rt, dirs, files in os.walk(temp_path):
            for f in files:
                file = os.path.join(temp_path, f)
                with open(file, encoding='utf8') as f:
                    line = f.readline()
                    while line:
                        row = line.strip().split(',')
                        words, cnt = row[0], int(row[1])
                        if words not in self.ngram_dict:
                            self.ngram_dict[words] = 0
                        self.ngram_dict[words] += cnt
                        line = f.readline()

        out_content = ''
        for k, v in self.ngram_dict.items():
            idf = round(math.log(1.0 * doc_cnt / v, 2), 5)
            out_content += k + ',' + str(idf) + '\n'

        with open(self.idf_path, 'w', encoding='utf8') as f:
            f.write(out_content)
        import shutil
        #shutil.rmtree(temp_path)
        print('The process of idf has done.')


    def load_idf(self):
        with open(self.idf_path, encoding='utf8') as f:
            line = f.readline()
            while line:
                row = line.strip().split(',')
                ngram_word = row[0]
                idf_weight = float(row[1])
                self.idf_dict[ngram_word] = idf_weight
                line = f.readline()


    def calculate_tfidf(self,documnet):
        if len(self.idf_dict) == 0:
            print('idf dict is empty, get the idf first!')
            sys.exit()
        tfidf_dict = {}
        words_dict = {}
        ngram_sum = 0
        for i in range(len(documnet)):
            sentence = documnet[i]
            for j in range(len(sentence) - 1):
                if sentence[j] in self.ignored or sentence[j + 1] in self.ignored:
                    continue
                ngram_word = ' '.join(sentence[j:j + self.ngram])
                ngram_sum += 1
                if ngram_word not in words_dict:
                    words_dict[ngram_word] = 0
                words_dict[ngram_word] += 1
        for k, v in words_dict.items():
            tf = v * 1.0 / ngram_sum
            idf = self.idf_dict[k]
            tfidf_value = tf * idf
            tfidf_dict[k] = tfidf_value

        return tfidf_dict

    def read_file(self, path):
        document = []
        with open(path, encoding='utf8', errors='ignore') as f:
            line = f.readline()
            while line:
                sentences = nltk.sent_tokenize(line.strip())
                sentences = [nltk.word_tokenize(s) for s in sentences]
                for sentence in sentences:
                    sentence = [word for word in sentence if word not in self.ignored]
                    document.append(sentence)
                line = f.readline()
        return document

    def find_keywords(self, document, topK):
        tfidf = self.calculate_tfidf(document)
        sorted_tfidf = sorted(tfidf.items(), key=lambda e:e[1], reverse=True)
        return [sorted_tfidf[i] for i in range(topK)]


if __name__ == '__main__':
    import time

    time_start = time.time()


    doc = Document('../input/wiki_head_10.txt')
    tfidf = TFIDF(
        documents=doc,
        ngram=1,
        stop_words_path='../input/stop_words.txt',
        idf_path='../output/idf.txt'
    )
    #tfidf.multi_pro_idf(process_num=2, p_doc_num=5)
    tfidf.load_idf()
    doc = tfidf.read_file('../input/wiki_test.txt')
    print(tfidf.find_keywords(doc, 10))
    time_end = time.time()
    print('time cost', time_end - time_start, 's')