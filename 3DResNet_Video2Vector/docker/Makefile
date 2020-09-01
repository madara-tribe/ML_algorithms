# https://qiita.com/matoken/items/664e7a7e8f31e8a46a60
MOV = cbt.mov
FIRST_T=0
END_T=5
IMGS=16

install:
	brew install ffmpeg

toimg:
	ffmpeg -i ${MOV} -ss ${FIRST_T} -t ${END_T} -r ${IMGS} -f image2 %06d.jpg
