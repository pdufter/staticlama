##############
# myenv mult
USERDIR="/mounts/work/philipp"
WORKDIR="${USERDIR}/staticlama-debug"
mkdir -p ${WORKDIR}

fasttextpath="/mounts/Users/cisintern/philipp/Dokumente/fastText-0.9.1/fasttext"
langs="en th ja de es fi tr ar ko he"


##############
# download mlama data
mkdir -p ${WORKDIR}/data/
wget http://cistern.cis.lmu.de/mlama/mlama1.1-all.zip -P ${WORKDIR}/data/
unzip ${WORKDIR}/data/mlama1.1-all.zip -d ${WORKDIR}/data/
mlamadata="${WORKDIR}/data/mlama1.1-all"


##############
# download wikipedias:
wikidir="${WORKDIR}/data/wiki"
cwd=$(pwd)
for lang in ${langs}
do
	# use https://github.com/attardi/wikiextractor
	mkdir -p ${wikidir}/wiki_${lang}
	cd ${wikidir}/wiki_${lang}
	wget http://download.wikimedia.org/${lang}wiki/latest/${lang}wiki-latest-pages-articles.xml.bz2
	/mounts/Users/cisintern/philipp/Dokumente/wikiextractor/WikiExtractor.py -cb 250K -o extracted ${lang}wiki-latest-pages-articles.xml.bz2
	find extracted -name '*bz2' -exec bunzip2 -c {} \; > wiki${lang}.xml
	rm -rf extracted
done
cd ${cwd}


##############
# preprocess wikipedias
for lang in ${langs} 
do
	python -m utils.sentence_tokenize \
	--infile ${wikidir}/wiki_${lang}/wiki${lang}.xml \
	--outfile ${wikidir}/wiki_${lang}/wiki${lang}-text.txt &
done

##############
# prepare corpora in parallel
for lang in ${langs} 
do

	mkdir -p ${WORKDIR}/corpora/${lang}
	mkdir -p ${WORKDIR}/vocab/${lang}

	# create vocabulary
	for vocabsize in 30000 120000 250000 500000 1000000
	do
		mkdir -p ${WORKDIR}/vocab/${lang}/wiki${lang}-text-${vocabsize} &&

		python -m utils.get_vocabulary \
		--infile ${USERDIR}/data/wiki/wiki_${lang}/wiki${lang}-text.txt \
		--outfolder ${WORKDIR}/vocab/${lang}/wiki${lang}-text-${vocabsize} \
		--vocabsize ${vocabsize} &&

		python -m utils.prepare_for_fasttext \
			--corpus ${USERDIR}/data/wiki/wiki_${lang}/wiki${lang}-text.txt \
			--vocab ${WORKDIR}/vocab/${lang}/wiki${lang}-text-${vocabsize} \
			--prefix "" \
			--outfile ${WORKDIR}/corpora/${lang}/wiki${lang}-text,wiki${lang}-text-${vocabsize}.txt &

	done

	# mbert vocab
	python -m utils.prepare_for_fasttext \
		--corpus ${USERDIR}/data/wiki/wiki_${lang}/wiki${lang}-text.txt \
		--vocab bert-base-multilingual-cased \
		--prefix "" \
		--outfile ${WORKDIR}/corpora/${lang}/wiki${lang}-text,bert-base-multilingual-cased.txt &


	# only for english: bert-base
	if [ ${lang} = "en" ]
	then
		python -m utils.prepare_for_fasttext \
		--corpus ${USERDIR}/data/wiki/wiki_${lang}/wiki${lang}-text.txt \
		--vocab bert-base-cased \
		--prefix "" \
		--outfile ${WORKDIR}/corpora/${lang}/wiki${lang}-text,bert-base-cased.txt &
	fi
done

##############
# run fasttext
for lang in ${langs} 
do

	mkdir -p ${WORKDIR}/embeddings/${lang}
	# create vocabulary
	#for vocabsize in 120000 250000 1000000
	for vocabsize in 30000 120000 250000 500000 1000000
	do

		# train monolingual spaces
		nice -n 19 ${fasttextpath} skipgram \
		-input ${WORKDIR}/corpora/${lang}/wiki${lang}-text,wiki${lang}-text-${vocabsize}.txt \
		-output ${WORKDIR}/embeddings/${lang}/wiki${lang}-text,wiki${lang}-text-${vocabsize} \
		-dim 300 \
		-thread 96 
	done


	nice -n 19 ${fasttextpath} skipgram \
	-input ${WORKDIR}/corpora/${lang}/wiki${lang}-text,bert-base-multilingual-cased.txt \
	-output ${WORKDIR}/embeddings/${lang}/wiki${lang}-text,bert-base-multilingual-cased \
	-dim 300 \
	-thread 96 

	# only for english: bert-base
	if [ ${lang} = "en" ]
	then
		nice -n 19 ${fasttextpath} skipgram \
		-input ${WORKDIR}/corpora/${lang}/wiki${lang}-text,bert-base-cased.txt \
		-output ${WORKDIR}/embeddings/${lang}/wiki${lang}-text,bert-base-cased \
		-dim 300 \
		-thread 96 
	fi
done

##############
# run staticlama
exid="debug"
# get the dataset e.g., from http://cistern.cis.lmu.de/mlama/
#idfilter="configs/UHN_uuids.json"
idfilter="all"
topn=5
for lang in ${langs} 
do
	mkdir -p "${WORKDIR}/results/${exid}"

	# mBERT
	python staticlama.py \
	--topn ${topn} \
	--data ${mlamadata} \
	--lang ${lang} \
	--details "${WORKDIR}/results/${exid}/${lang},bert-base-multilingual-cased.json" \
	--summary "${WORKDIR}/results/${exid}.txt" \
	--lm "bert-base-multilingual-cased" \
	--relations "all" \
	--idfilter ${idfilter} \
	--prefix "lm"

	# BERT-BASE
	if [ ${lang} = "en" ]
	then
		python staticlama.py \
		--topn ${topn} \
		--data ${mlamadata} \
		--lang ${lang} \
		--details "${WORKDIR}/results/${exid}/${lang},bert-base-cased.json" \
		--summary "${WORKDIR}/results/${exid}.txt" \
		--lm "bert-base-cased" \
		--relations "all" \
		--idfilter ${idfilter} \
		--prefix "lm"
	fi
	# STATIC fastText
	for vocabsize in 30000 120000 250000 500000 1000000
	do
		python staticlama.py \
		--topn ${topn} \
		--data ${mlamadata} \
		--lang ${lang} \
		--details "${WORKDIR}/results/${exid}/wiki${lang}-text,wiki${lang}-text-${vocabsize}.json" \
		--summary "${WORKDIR}/results/${exid}.txt" \
		--embeddings "${WORKDIR}/embeddings/${lang}/wiki${lang}-text,wiki${lang}-text-${vocabsize}.vec" \
		--vocab "${WORKDIR}/vocab/${lang}/wiki${lang}-text-${vocabsize}" \
		--relations "all" \
		--similaritymeasure "cosine" \
		--idfilter ${idfilter} \
		--prefix "static"
	done

	# STATIC mBERT Vocab
	python staticlama.py \
	--topn ${topn} \
	--data ${mlamadata} \
	--lang ${lang} \
	--details "${WORKDIR}/results/${exid}/wiki${lang}-text,bert-base-multilingual-cased.json" \
	--summary "${WORKDIR}/results/${exid}.txt" \
	--embeddings "${WORKDIR}/embeddings/${lang}/wiki${lang}-text,bert-base-multilingual-cased.vec" \
	--vocab "bert-base-multilingual-cased" \
	--relations "all" \
	--similaritymeasure "cosine" \
	--idfilter ${idfilter} \
	--prefix "static"

	# STATIC BERT-BASE
	if [ ${lang} = "en" ]
	then
		python staticlama.py \
		--topn ${topn} \
		--data ${mlamadata} \
		--lang ${lang} \
		--details "${WORKDIR}/results/${exid}/wiki${lang}-text,bert-base-cased.json" \
		--summary "${WORKDIR}/results/${exid}.txt" \
		--embeddings "${WORKDIR}/embeddings/${lang}/wiki${lang}-text,bert-base-cased.vec" \
		--vocab "bert-base-cased" \
		--relations "all" \
		--similaritymeasure "cosine" \
		--idfilter ${idfilter} \
		--prefix "static"
	fi

	# MAJORITY
	python staticlama.py \
	--topn ${topn} \
	--data ${mlamadata} \
	--lang ${lang} \
	--details "${WORKDIR}/results/${exid}/${lang},majority.json" \
	--summary "${WORKDIR}/results/${exid}.txt" \
	--majority \
	--relations "all" \
	--idfilter ${idfilter} \
	--prefix "majority"

	# mBERT STATIC LAYER
	for layer in {0..12}
	do
		python staticlama.py \
		--topn ${topn} \
		--data ${mlamadata} \
		--lang ${lang} \
		--details "${WORKDIR}/results/${exid}/${lang},bert-base-multilingual-cased-lmstatic-${layer}.json" \
		--summary "${WORKDIR}/results/${exid}.txt" \
		--lm "bert-base-multilingual-cased" \
		--lmstatic ${layer} \
		--relations "all" \
		--idfilter ${idfilter} \
		--prefix "lmstatic-${layer}"
	done

	# BERT STATIC LAYER
	if [ ${lang} = "en" ]
	then
		for layer in {0..12}
		do
			python staticlama.py \
			--topn ${topn} \
			--data ${mlamadata} \
			--lang ${lang} \
			--details "${WORKDIR}/results/${exid}/${lang},bert-base-cased-lmtatic-${layer}.json" \
			--summary "${WORKDIR}/results/${exid}.txt" \
			--lm "bert-base-cased" \
			--lmstatic ${layer} \
			--relations "all" \
			--idfilter ${idfilter} \
			--prefix "lmstatic-${layer}"
		done
	fi
done
