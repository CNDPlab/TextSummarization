cat Datas/Sougou/news_sohusite_xml.dat | iconv -f gb18030 -t utf-8 -c | grep -e "<content>" -e "<contenttitle>"  > Datas/raw/corpus.txt