cat raw/news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep -e "<content>" -e "<contenttitle>"  > raw/corpus.txt
