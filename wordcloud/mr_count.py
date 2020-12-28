import mrjob,os
from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import heapq

WORD_RE = re.compile(r"[\w']+")

TOPN=100
class MRMostUsedWord(MRJob):

    def mapper_get_words(self, _, line):
        for word in WORD_RE.findall(line):
            yield (word.lower(), 1)
        

    def combiner_count_words(self, word, counts):
        yield word, sum(counts)


    def reducer_init(self):
        self.heap = []


    def reducer_count_words(self, word, counts):
        heapq.heappush(self.heap,(sum(counts),word))
        # we only keep top 10 words in the heap of each reducer
        if len(self.heap) > TOPN:
            heapq.heappop(self.heap)

    # get top 10 words in each reducer
    def reducer_final(self):
        for (count,word) in self.heap:
            yield (word,count)

    # Step 2 — The global TopN needs to run.
    # form the count_word pairs
    def globalTopN_mapper(self,word,count):
        yield "Top"+str(TOPN), (count,word)

    # get the global top10 words
    def globalTopN_reducer(self,_,countsAndWords):
        for countAndWord in heapq.nlargest(TOPN,countsAndWords):
            yield countAndWord
        
    # Steps causes mrjob to run multiple jobs.
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   combiner=self.combiner_count_words,
                   reducer_init=self.reducer_init,
                   reducer=self.reducer_count_words,
                   reducer_final=self.reducer_final
                   ),

            MRStep(mapper=self.globalTopN_mapper,
                   reducer=self.globalTopN_reducer) ]

if __name__=="__main__":
    MRMostUsedWord.run()