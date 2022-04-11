# Developed By Ghidorah (2019-2020)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import state_union
from nltk.corpus import stopwords

from bs4 import BeautifulSoup as bs

from datetime import datetime

from requests import get

import numpy as np
import mysql.connector
import threading
import keyboard
import pickle
import click
import nltk
import time
import aiml
import sys
import re
import os

# Global Variable
AutoChat = 1
CountReceive = 0
CountReplied = 0
Now = datetime.now()
Tahun = 2019
Praktikum = []


# Koneksi Database MySql
conn = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="",
	database="si_lab"
)
mycursor = conn.cursor()


# Define XPATH
xpath_unread_msg = "//*[@class='l7jjieqr cfzgl7ar ei5e7seu h0viaqh7 tpmajp1w c0uhu3dl riy2oczp dsh4tgtl sy6s5v3r gz7w46tb lyutrhe2 qfejxiq4 fewfhwl7 ovhn1urg ap18qm3b ikwl5qvt j90th5db aumms1qt']"
xpath_user_name = '//*[@class="ldL67 _3sh5K"]//*[@class="_21nHd"]'
xpath_msg_box = '//*[@id="main"]/footer/div[1]/div/span[2]/div/div[2]/div[1]/div/div[2]'
xpath_send_button = '//*[@id="main"]/footer/div[1]/div/span[2]/div/div[2]/div[2]/button'
xpath_title = "//span[@title = '{}']"


# Brain Preparation
kernel = aiml.Kernel()
try:
	if os.path.isfile("brain/core_brain.brn"):
	    kernel.bootstrap(brainFile = "brain/core_brain.brn")
	else:
		kernel.learn("brain/sheet_greetings.aiml")
		kernel.learn("brain/sheet_chitchat.aiml")
		kernel.learn("brain/sheet_insult.aiml")
		kernel.learn("brain/sheet_about.aiml")
		kernel.learn("brain/sheet_query.aiml")
		kernel.learn("brain/commands_master.aiml")
		kernel.learn("brain/commands_system.aiml")
		kernel.saveBrain("brain/core_brain.brn")
	print("[BRAIN LOAD SUCCES]")
except Exception as e:
	print("[BRAIN LOAD FAILED]")


# NLP Preparation
train = open("assets/1-ind-training.txt","r")
ner_train = train.read()
stemmer = StemmerFactory().create_stemmer()
stop_words = ['sementara', 'berkehendak', 'tak', 'ia', 'sesekali', 'hanya', 'tandasnya', 'mulai', 'dimulainya', 'penting', 'kapanpun', 'diibaratkannya', 'dia', 'ingat-ingat', 'telah', 'didatangkan', 'dilalui', 'terbanyak', 'pula', 'sama-sama', 'berarti', 'mana', 'itu', 'dituturkan', 'beginilah', 'seterusnya', 'sebagainya', 'mengerjakan', 'bakalan', 'sekali', 'sebenarnya', 'lamanya', 'diri', 'perlukah', 'sebab', 'bukan', 'kepadanya', 'tapi', 'termasuk', 'tinggi', 'memperbuat', 'kami', 'kitalah', 'memberi', 'dimungkinkan', 'pernah', 'keterlaluan', 'menanyai', 'mengapa', 'apatah', 'makanya', 'pertanyaan', 'ucapnya', 'ibaratnya', 'diperlukannya', 'ditunjukkan', 'berapakah', 'diberikannya', 'itu', 'jawabnya', 'semaunya', 'cuma', 'dikerjakan', 'akhirnya', 'cukup', 'dari', 'datang', 'atas', 'nyatanya', 'saya', 'terhadap', 'kemungkinannya', 'dimulai', 'sebagai', 'mempertanyakan', 'sejak', 'seolah-olah', 'dituturkannya', 'ditanyakan', 'di', 'karenanya', 'mempunyai', 'menunjuki', 'itulah', 'sewaktu', 'inilah', 'lanjutnya', 'selaku', 'bagaimanapun', 'kapan', 'seketika', 'berikut', 'bakal', 'mengibaratkan', 'sini', 'masalah', 'setidak-tidaknya', 'apa', 'tandas', 'kalian', 'antaranya', 'belum', 'dijawab', 'kala', 'setiba', 'demikian', 'terlihat', 'dahulu', 'kalaulah', 'dimaksudnya', 'rasa', 'dilakukan', 'memulai', 'walaupun', 'para', 'semisal', 'asalkan', 'bersiap-siap', 'tentunya', 'begini', 'dimaksudkannya', 'seorang', 'enggak', 'gak', 'bolehkah', 'ibu', 'tengah', 'mendatang', 'jadinya', 'tentu', 'bersiap', 'disampaikan', 'maupun', 'sesudahnya', 'bagaikan', 'mengucapkan', 'mengira', 'benarlah', 'mulailah', 'bersama', 'jawab', 'seolah', 'lagi', 'diingatkan', 'jelasnya', 'adalah', 'disini', 'bagaimana', 'jelaslah', 'bulan', 'tanpa', 'beberapa', 'mendapat', 'jangankan', 'sebanyak', 'selalu', 'melihatnya', 'se', 'mengucapkannya', 'bisa', 'bagian', 'bilakah', 'entahlah', 'sebelum', 'ibaratkan', 'setiap', 'rata', 'dini', 'biasa', 'keinginan', 'terdapat', 'segalanya', 'berikutnya', 'berlebihan', 'bawah', 'menjawab', 'akhiri', 'diberikan', 'sama', 'terjadinya', 'bukannya', 'inikah', 'dengan', 'ditunjukkannya', 'persoalan', 'mengakhiri', 'semasih', 'sih', 'bermacam-macam', 'selain', 'pihak', 'sebaik-baiknya', 'malah', 'kira-kira', 'menghendaki', 'nah', 'panjang', 'sangatlah', 'didapat', 'lainnya', 'sebisanya', 'misalkan', 'menanyakan', 'berjumlah', 'dijelaskannya', 'jangan', 'dikatakannya', 'kasus', 'teringat-ingat', 'terutama', 'memang', 'perlu', 'akhir', 'adanya', 'begitukah', 'tanyanya', 'sebetulnya', 'beginian', 'bertanya-tanya', 'kemudian', 'padahal', 'sehingga', 'sekitar', 'wong', 'ucap', 'bolehlah', 'dirinya', 'sekadar', 'nanti', 'seringnya', 'meskipun', 'jelaskan', 'lagian', 'melalui', 'sedemikian', 'menyeluruh', 'berturut-turut', 'kemungkinan', 'dong', 'balik', 'hal', 'berada', 'kira', 'tanyakan', 'pada', 'sepertinya', 'berdatangan', 'dialah', 'kok', 'mungkinkah', 'hingga', 'diakhiri', 'ikut', 'pantas', 'tepat', 'dimintai', 'sendirinya', 'bapak', 'pak', 'tegasnya', 'ketika', 'khususnya', 'akulah', 'semata-mata', 'diingat', 'amat', 'dijelaskan', 'tiba', 'terjadi', 'terlalu', 'sampai-sampai', 'dikatakan', 'adapun', 'berlainan', 'perlunya', 'seluruhnya', 'tegas', 'saat', 'kinilah', 'memperkirakan', 'merekalah', 'meski', 'anda', 'semampu', 'jawaban', 'yakni', 'tidak', 'soalnya', 'dimulailah', 'ditunjuknya', 'paling', 'secara', 'teringat', 'awal', 'tadinya', 'kita', 'semata', 'melainkan', 'enggaknya', 'menginginkan', 'semampunya', 'agar', 'berikan', 'membuat', 'keadaan', 'cukuplah', 'diucapkan', 'mengenai', 'seusai', 'masihkah', 'meminta', 'andalah', 'belakang', 'lewat', 'supaya', 'melihat', 'kelihatan', 'pasti', 'mengingat', 'kamulah', 'jika', 'yaitu', 'menanya', 'pentingnya', 'berakhirlah', 'dipergunakan', 'mengibaratkannya', 'atau', 'semisalnya', 'bersama-sama', 'hanyalah', 'diketahui', 'sekaligus', 'mulanya', 'siapapun', 'disebut', 'sajalah', 'berturut', 'menegaskan', 'diungkapkan', 'rasanya', 'agaknya', 'ialah', 'sedikit', 'ungkapnya', 'sebegitu', 'walau', 'dipertanyakan', 'kelihatannya', 'menanti-nanti', 'agak', 'nyaris', 'rupanya', 'wahai', 'hendak', 'seingat', 'dalam', 'menyebutkan', 'berapapun', 'tersampaikan', 'sekadarnya', 'tertuju', 'memintakan', 'kembali', 'suatu', 'bagi', 'diperbuatnya', 'hari', 'sudahkah', 'tetap', 'terdahulu', 'mampu', 'terlebih', 'sesuatunya', 'nantinya', 'keseluruhan', 'terus', 'sebutnya', 'keduanya', 'harusnya', 'bila', 'sempat', 'bahwasanya', 'seseorang', 'misalnya', 'bung', 'makin', 'mungkin', 'begitulah', 'dipunyai', 'waktu', 'hendaklah', 'sendiri', 'diberi', 'tahun', 'semasa', 'saling', 'karena', 'mengatakannya', 'berlangsung', 'turut', 'sebabnya', 'sesaat', 'bukanlah', 'sekurangnya', 'diinginkan', 'haruslah', 'amatlah', 'kurang', 'bahkan', 'sepantasnya', 'tertentu', 'menyampaikan', 'lama', 'sayalah', 'setelah', 'memisalkan', 'sudahlah', 'kecil', 'akan', 'tentulah', 'banyak', 'seperlunya', 'dibuat', 'sedikitnya', 'benar', 'itukah', 'ungkap', 'baik', 'depan', 'berujar', 'pastilah', 'inginkah', 'bermacam', 'dimaksud', 'entah', 'mendatangi', 'dipersoalkan', 'seenaknya', 'kenapa', 'kalau', 'betulkah', 'tidakkah', 'setempat', 'oleh', 'sekali-kali', 'sesama', 'menambahkan', 'secukupnya', 'gunakan', 'dilihat', 'tersebut', 'sesampai', 'bertutur', 'kalaupun', 'seberapa', 'serupa', 'sangat', 'sinilah', 'keseluruhannya', 'minta', 'diminta', 'bertanya', 'misal', 'caranya', 'ataupun', 'tentang', 'sambil', 'masih', 'dekat', 'mempergunakan', 'tunjuk', 'daripada', 'macam', 'segala', 'ada', 'janganlah', 'dikira', 'menuju', 'menaiki', 'ditunjuki', 'keluar', 'menurut', 'ingat', 'segera', 'sekecil', 'berawal', 'disinilah', 'pukul', 'terdiri', 'seperti', 'diperlukan', 'dan', 'berakhirnya', 'semuanya', 'ini', 'asal', 'hendaknya', 'kebetulan', 'betul', 'begitu', 'masa', 'boleh', 'besar', 'sendirian', 'demi', 'jumlah', 'waktunya', 'sedangkan', 'mau', 'selanjutnya', 'memungkinkan', 'diperlihatkan', 'sebagian', 'punya', 'diakhirinya', 'belumlah', 'sebutlah', 'terhadapnya', 'manakala', 'dapat', 'tetapi', 'menunjuknya', 'sampaikan', 'sekarang', 'ditambahkan', 'menyangkut', 'masing-masing', 'mula', 'terasa', 'lanjut', 'sebesar', 'setinggi', 'lah', 'dimaksudkan', 'ternyata', 'sekitarnya', 'jikalau', 'terkira', 'beri', 'cukupkah', 'ke', 'tambah', 'semakin', 'berlalu', 'setengah', 'menantikan', 'per', 'tahu', 'olehnya', 'semacam', 'berkeinginan', 'soal', 'berupa', 'memerlukan', 'berkenaan', 'mampukah', 'berkali-kali', 'terjadilah', 'usai', 'meyakinkan', 'akankah', 'umum', 'dimisalkan', 'saatnya', 'bisakah', 'bagai', 'katakan', 'katakanlah', 'lain', 'mendatangkan', 'sebegini', 'sekalipun', 'sampai', 'dipastikan', 'mirip', 'kapankah', 'bekerja', 'ditanyai', 'diperbuat', 'sebaliknya', 'sepanjang', 'antar', 'sejauh', 'kepada', 'mereka', 'jadi', 'ujar', 'tutur', 'naik', 'bagaimanakah', 'terakhir', 'ingin', 'selama-lamanya', 'sebagaimana', 'lalu', 'apaan', 'disebutkannya', 'pihaknya', 'wah', 'siap', 'begitupun', 'bermaksud', 'tersebutlah', 'mempersoalkan', 'diantaranya', 'ditanya', 'ditujukan', 'tambahnya', 'selama', 'meyakini', 'sebaiknya', 'tempat', 'tiap', 'padanya', 'selamanya', 'diibaratkan', 'apakah', 'manalagi', 'ditunjuk', 'yakin', 'bermula', 'justru', 'disebutkan', 'diucapkannya', 'lebih', 'untuk', 'katanya', 'sekiranya', 'memperlihatkan', 'mendapatkan', 'menggunakan', 'buat', 'dibuatnya', 'diketahuinya', 'harus', 'jadilah', 'kamilah', 'menanti', 'menjadi', 'apalagi', 'tampak', 'bukankah', 'memberikan', 'sering', 'setibanya', 'inginkan', 'sesegera', 'juga', 'masing', 'menjelaskan', 'tidaklah', 'berakhir', 'belakangan', 'sesuatu', 'sebelumnya', 'benarkah', 'beginikah', 'jelas', 'mengatakan', 'semula', 'toh', 'berapa', 'cara', 'dikarenakan', 'ditandaskan', 'serta', 'diperkirakan', 'tampaknya', 'merasa', 'menyiapkan', 'siapakah', 'diantara', 'dulu', 'berbagai', 'antara', 'awalnya', 'digunakan', 'seluruh', 'mengingatkan', 'umumnya', 'kelamaan', 'kini', 'memihak', 'sejenak', 'pertama-tama', 'menunjuk', 'ditegaskan', 'masalahnya', 'waduh', 'berapalah', 'sana', 'kiranya', 'usah', 'ujarnya', 'sebut', 'mengetahui', 'sejumlah', 'biasanya', 'ataukah', 'luar', 'sela', 'memastikan', 'kata', 'bu', 'sebuah', 'tadi', 'guna', 'menandaskan', 'maka', 'kesampaian', 'sepantasnyalah', 'sesudah', 'baru', 'malahan', 'merupakan', 'sedang', 'tiba-tiba', 'jauh', 'apabila', 'menunjukkan', 'demikianlah', 'pun', 'sekalian', 'melakukan', 'saja', 'tuturnya', 'seharusnya', 'mengungkapkan', 'bahwa', 'sebaik', 'sekurang-kurangnya', 'sudah', 'mempersiapkan', 'hampir', 'jumlahnya', 'menyatakan', 'artinya', 'setidaknya', 'ibarat', 'percuma', 'menuturkan', 'kan', 'sepihak', 'namun', 'pertanyakan', 'berkata']
custome_sent_tokenizer = PunktSentenceTokenizer(ner_train)
with open("assets/lex_id.pickle", "rb") as fh:
    lex = pickle.load(fh)
tagger = PerceptronTagger(load="assets/averaged_perceptron_tagger_id.pickle")


# Chrome Driver Preparation
driver = webdriver.Chrome()
driver.maximize_window()
driver.get("https://web.whatsapp.com/")
input("Tekan Enter setelah Selesai Scan QR Code")


def greetings():
	print("")
	print("------------------------")
	print("-  CHATBOT BY DIMASFR  -")
	print("-                      -")
	print("------------------------")
	for i in range(101):
	    sys.stdout.write('\r')
	    sys.stdout.write("%-10s %d%%" % ("[LOADING] Please wait..", 1*i))
	    sys.stdout.flush()
	    time.sleep(0.05)
	print("")


def init():
	global Praktikum
	try:
		mycursor.execute("SELECT * FROM praktikum")
		Praktikum = mycursor.fetchall()
	except Exception as e:
		print("[ERROR] GET PRAKTIKUM INFORMATION : FAILED")
		print(str(e))


def ind_pos(sent, lex=lex, cm='⊝'): #Proses POS Tagging Indonesia
    sf1 = ["nya", "ku", "kau", "mu"]
    sf2 = ["lah", "kah"]
    pf =  ["ku", "kau", "ke", "se"]
    redup = re.compile("^(.*)-\\1$", flags=re.IGNORECASE)
    rsf2  = re.compile("^(.*)({})$".format("|".join(sf2)), flags=re.IGNORECASE)
    rsf12 = re.compile("^(.*?)(-)?({})({})?$".format("|".join(sf1),"|".join(sf2)), flags=re.IGNORECASE)
    rpf   = re.compile("^({})(.*)$".format("|".join(pf)), flags=re.IGNORECASE)
    raf12 = re.compile("^({})(.*?)(-)?({})({})?$".format("|".join(pf), "|".join(sf1),"|".join(sf2)), flags=re.IGNORECASE)
    toks = []
    for word in nltk.word_tokenize(sent):
        rm = redup.match(word)
        if word.lower() in lex['all']:
            toks.append(word)
        elif rm and  rm.group(1).lower() in lex['all']:
            toks.append(word)
        else:
            m = rsf2.match(word)
            if m:
                toks.append(m.group(1))
                toks.append("{}{}".format(cm,m.group(2)))
                continue
            m = rpf.match(word)
            if m:
                toks.append("{}{}".format(m.group(1),cm))
                toks.append(m.group(2))
                continue
            m = rsf12.match(word)
            if m:
                toks.append(m.group(1))
                if m.group(2):
                    toks.append("{}{}".format(m.group(2),m.group(3)))
                else:
                    toks.append("{}{}".format(cm,m.group(3)))
                if m.group(4):
                    toks.append("{}{}".format(cm,m.group(4)))
                continue
            m = raf12.match(word)
            if m:
                toks.append("{}{}".format(m.group(1),cm))
                toks.append(m.group(2))
                if m.group(3):
                    toks.append("{}{}".format(m.group(3),m.group(4)))
                else:
                    toks.append("{}{}".format(cm,m.group(4)))
                if m.group(4):
                    toks.append("{}{}".format(cm,m.group(5)))
                continue
            toks.append(word)
    
    return toks


def ner(words,session): #Proses Named Entity Recognition
	file_a = open("log/"+session+'.txt',"a")
	for word in custome_sent_tokenizer.tokenize(words):
		for chunk in nltk.ne_chunk(tagger.tag(ind_pos(word,cm='-'))):
			if hasattr(chunk, 'label'):
				print(chunk.label(), ' '.join(c[0] for c in chunk))
				label = chunk.label()
				entity = ' '.join(c[0] for c in chunk)
				file_a.write(label + " " + entity + " ")
	file_a.close()


def preprocessing(words): #Proses konversi kalimat
	katadasar = stemmer.stem(words)
	token = word_tokenize(katadasar)
	filered_sentence = [x for x in token if x not in stop_words]
	de_token = TreebankWordDetokenizer().detokenize(filered_sentence)
	return de_token


def input_single_nilai(kode_praktikum,nim,tahun,nilai,ket): #Input single data ke Nilai
	try:
		sql = "UPDATE data_nilai SET "+ ket +" = %s WHERE kode_praktikum = %s AND nim_mahasiswa = %s AND tahun = %s"
		val = (nilai, kode_praktikum, nim, tahun)
		mycursor.execute(sql, val)
		conn.commit()
		return "Data berhasil di update!"
	except Exception as e:
		print("[ERROR] SINGLE UPDATE NILAI : FAILED")
		print(str(e))
		return "Data gagal di update!"


def input_batch_nilai(kode_praktikum,nim,tahun,nilai,ket): #Input batch data ke Nilai
	try:
		sql = "UPDATE data_nilai SET "+ ket +" = %s WHERE kode_praktikum = %s AND nim_mahasiswa = %s AND tahun = %s"
		val = (nilai, kode_praktikum, nim, tahun)
		mycursor.execute(sql, val)
		conn.commit()
		return "Data "+ nim +" berhasil di update!"
	except Exception as e:
		print("[ERROR] SINGLE UPDATE NILAI : FAILED")
		print(str(e))
		return "Data gagal di update!"


def view_single_nilai(kode_praktikum,nim,tahun,ket): #Lihat single data di Nilai
	try:
		mycursor.execute("SELECT "+ket+" FROM data_nilai WHERE kode_praktikum = '"+str(kode_praktikum)+"' AND nim_mahasiswa = "+str(nim)+" AND tahun = "+str(tahun)+"")
		myresult = mycursor.fetchone()
		return myresult
	except Exception as e:
		print("[ERROR] SINGLE SELECT NILAI : FAILED")
		print(str(e))


def view_batch_nilai(kode_praktikum,nim,tahun): #Lihat batch data di Nilai
	try:
		mycursor.execute("SELECT nt1, nt2, nt3, nt4, nt5, nt6, nt7, nt8, nt9, nt10 FROM data_nilai WHERE kode_praktikum = '"+str(kode_praktikum)+"' AND nim_mahasiswa = "+str(nim)+" AND tahun = "+str(tahun)+"")
		myresult = mycursor.fetchone()	
		return myresult
	except Exception as e:
		print("[ERROR] BATCH SELECT NILAI : FAILED")
		print(str(e))


def view_single_nim(nim): #Cek Nama Mahasiswa
	try:
		mycursor.execute("SELECT nama_mahasiswa FROM data_mahasiswa WHERE nim_mahasiswa = '"+str(nim)+"'")
		myresult = mycursor.fetchone()
		return myresult
	except Exception as e:
		print("[ERROR] SEARCH NAME : FAILED")
		print(str(e))
		return False


def view_single_nip(nama): #Cek NIP Dosen
	try:
		mycursor.execute("SELECT nip_dosen,nama_dosen FROM data_dosen WHERE nama_dosen LIKE %s LIMIT 1", ("%"+nama+"%",))
		myresult = mycursor.fetchone()
		return myresult
	except Exception as e:
		print("[ERROR] SEARCH NIP DOSEN : FAILED")
		print(str(e))
		return False


def validating_aslab(no): #Validasi Hak Akses Aslab
	no = str(no)
	no = no.replace(" ","")
	no = no.replace("-","")
	no = no.replace("+","")
	try:
		mycursor.execute("SELECT a.laboratorium FROM data_aslab a JOIN admin b ON (a.nim_aslab = b.nim) WHERE b.no_telp = '"+no+"'")
		myresult = mycursor.fetchone()
		return myresult
	except Exception as e:
		print("[ERROR] VALIDASI HAK AKSES ASLAB : FAILED")
		print(str(e))
		return False


def send_function(balasan): #Proses membalas pesan
	try:
		driver.find_element(By.XPATH, xpath_msg_box).send_keys(balasan)
		time.sleep(1)
		driver.find_element(By.XPATH, xpath_send_button).click()
	except Exception as e:
		print("[ERROR] SEND MESSAGE : FAILED")
		print(str(e))


def write_log(session,waktu,pesan,stts): #Proses penulisan log di .html
	file = open("log/"+session+'.html',"a",encoding="utf-8")
	if stts == 1:
		file.write("<div class='msgln'>" + waktu + " <b>" + session + "</b> : " + pesan +"<br></div>\n")
	else:
		file.write("<div class='msgln'>" + waktu + " <b>Dira</b> : " + pesan +"<br></div>\n")
	file.close()


def validating_nim(session): #Mengecek Nim di file .txt
	file_a = open("log/"+session+'.txt',"a")
	file_r = open("log/"+session+'.txt',"r")
	content = file_r.read()
	file_a.close()
	file_r.close()
	if "NIM" in content:
		token = word_tokenize(content)
		nim = token[1]
		return nim
	else:
		return False
	

def validating_nama(session): #Mengecek Nama di file .txt
	file_a = open("log/"+session+'.txt',"a")
	file_r = open("log/"+session+'.txt',"r")
	content = file_r.read()
	file_a.close()
	file_r.close()
	if "NAMA" in content:
		token = word_tokenize(content)
		nama = token[3]
		return nama
	else:
		return False


def validating_role(session): #Mengecek Role di file .txt
	file_a = open("log/"+session+'.txt',"a")
	file_r = open("log/"+session+'.txt',"r")
	content = file_r.read()
	file_a.close()
	file_r.close()
	if "ROLE" in content:
		token = word_tokenize(content)
		role = token[5]
		return role
	else:
		return False


def register_nim(session): #Mengisi Nim di file .txt
	temp = []
	get_nim = kernel.respond("SYSTEM GET NIM",session)
	if "EXPOSE NIM" in get_nim:
		temp = word_tokenize(get_nim)
		if len(temp)>2:
			nim = temp[2]
			file_a = open("log/"+session+'.txt',"a")
			file_a.write("NIM " + nim + " ")
			file_a.close()
			return True
		else:
			return False

def register_nama(session): #Mengisi Nama di file .txt
	temp = []
	get_nama = kernel.respond("SYSTEM GET NAME",session)
	if "EXPOSE NAME" in get_nama:
		temp = word_tokenize(get_nama)
		if len(temp)>2:
			nama = temp[2]
			file_a = open("log/"+session+'.txt',"a")
			file_a.write("NAMA " + nama + " ")
			file_a.close()
			return True
		else:
			return False


def auto_commands(word,session,waktu): #Perintah otomatis ketika user tidak dikenali
	order = kernel.respond(word,session)
	send_function(order)
	print("[SENT MESSAGE] (" + waktu + ") " + order)
	file = open("log/"+session+'.html',"a",encoding="utf-8")
	file.write("<div class='msgln'>" + waktu + " <b>Dira</b> : " + order +"<br></div>\n")
	file.close()


def check_xpath(xpath): #Mengecek Xpath tersedia atau tidak
    try:
       driver.find_element(By.XPATH, xpath)
    except NoSuchElementException:
        return False
    return True


def recheck(words,session,role): #Pengecekan keywords
	global AutoChat, CountReceive, CountReplied, Tahun, Praktikum
	data_temp = []
	temp1 = ""
	temp2 = ""
	balasan = ""
	if "MASTER :" in words: # Master Authority
		if role == "MASTER":
			if "EXIT" in words:
				print("[PROGRAM IS CLOSING]")
				exit()
			elif "PAUSE" in words:
				print("[PROGRAM IS PAUSED]")
				balasan = "Pausing Program..."
				AutoChat = 0
			elif "RESUME" in words:
				print("[PROGRAM IS RESUMING]")
				balasan = "Resuming Program..."
				AutoChat = 1
			elif "STATUS" in words:
				if AutoChat == 1:
					balasan = "Program is Online \nPesan di Terima " + str(CountReceive) + " \nPesan di Kirim " + str(CountReplied)
				else:
					balasan = "Program is Paused \nPesan di Terima " + str(CountReceive) + " \nPesan di Kirim " + str(CountReplied)
		else:
			balasan = "Akses Ditolak!"

	elif "SYSTEM :" in words:
		if "HELP" in words:
			balasan = "HELP : Menampilkan daftar fungsi yang dimiliki ChatBot"
			# balasan = str(open('log/help.txt').read().split('\n'))

	elif "QUERY" in words: # Accessing Database
		if "NILAI" in words:
			for x in word_tokenize(words):
				data_temp.append(x)
			for x in range(len(Praktikum)):
				if (Praktikum[x][2] != '' and Praktikum[x][2] in words.lower()) or (Praktikum[x][3] != '' and Praktikum[x][3] in words.lower()) or (Praktikum[x][4] != '' and Praktikum[x][4] in words.lower()):
					data_temp[4] = Praktikum[x][0]
					temp1 = Praktikum[x][5]
					temp2 = Praktikum[x][1]

			if "VIEWS" in words:
				if "SINGLE" in words:
					balasan = view_single_nilai(data_temp[4],data_temp[5],Tahun,data_temp[6])
					balasan = str(balasan)
					if balasan == "(0.0,)":
						balasan = "Mohon maaf, Sepertinya tugas anda belum di cek"
					elif balasan == "None":
						balasan = "Mohon maaf, Data yang anda minta tidak ditemukan"
					else:
						balasan = "Datamu sudah masuk ke dalam database"
				if "BATCH" in words:
					try:
						temp1 = "Informasi Nilai "+data_temp[5]+" Praktikum "+temp2+" :\n"
						balasan = view_batch_nilai(data_temp[4],data_temp[5],Tahun)
						for x in range(len(balasan)):
							temp1 += str(x+1)+". ["+"\u2713"+"] " if float(balasan[x]) != 0.0 else str(x+1)+". ["+"\u2715"+"] "
						balasan = temp1
					except Exception as e:
						balasan = "Data yang anda minta gagal di Load!"


			elif "UPDATES" in words:
				if role is not False:
					if role == temp1 or role == "MASTER":
						if (float(data_temp[7]) >= 0 and float(data_temp[7]) <= 100):
							if "SINGLE" in words:
								balasan = input_single_nilai(data_temp[4],data_temp[5],Tahun,data_temp[7],data_temp[6])
								balasan = str(balasan)
							if "BATCH" in words:
								balasan = input_batch_nilai(data_temp[4],data_temp[5],Tahun,data_temp[7],data_temp[6])
								balasan = str(balasan)
						else:
							balasan = "Nilai yang anda masukkan tidak sesuai standard nilai yang telah ditentukan."
					else:
						balasan = "Anda tidak memiliki Akses data Lab. " + temp1
				else:
					balasan = "Akses ditolak!"

		elif "NIP" in words:
			for x in word_tokenize(words):
				data_temp.append(x)

			if "VIEWS" in words:
				if "SINGLE" in words:
					try:
						balasan = view_single_nip(data_temp[4])
						balasan = "NIP : "+ balasan[0] + ". Nama Lengkap : " + balasan[1]
					except Exception as e:
						balasan = "Data tidak ditemukan"

	else:
		balasan = words

	return balasan


def sub(session,waktu,pesan,stts): #Program utama - sub
	global CountReplied, CountReceive
	current_time = Now.strftime("%H:%M")
	getValNim = validating_nim(session)
	getValNam = validating_nama(session)
	getValRol = validating_role(session)
	sessionData = kernel.getSessionData(session)
	if getValNim:
		kernel.setPredicate("nim_user", getValNim, session)
		if stts == 1:
			print("[USER NIM] "+ getValNim)
		if view_single_nim(getValNim) is False or view_single_nim(getValNim) is None:
			pesan = "SYSTEM FAILED FIND USER"
	if getValNam:
		kernel.setPredicate("nama_user", getValNam, session)
		if stts == 1:
			print("[USER NAME] "+ getValNam)
	if getValRol:
		if validating_aslab(session):
			if stts == 1:
				print("[ROLE] ASLAB "+ getValRol)
		else:
			getValRol = False
	if getValNam is not False and getValNim is not False:
		ner(pesan,session)
	else:
		if stts == 1:
			print("[USER NOT RECOGNIZED] No. Contact : "+ session)

	print("[READ MESSAGE] (" + waktu + ") " + pesan)

	write_log(session,waktu,pesan,1)

	CountReceive += 1

	for word in custome_sent_tokenizer.tokenize(pesan):
		CountReplied += 1
		counter = 0
		mark = 0
		while mark == 0:
			respon = kernel.respond(word,session)
			if respon == "NONE" and counter == 0:
				word = preprocessing(word)
				counter = 1
			elif (respon == "NONE" or respon == "") and counter == 1:
				word = "PATTERN NOT FOUND"
				counter = 2
			else:
				mark = 1

		balasan = recheck(respon,session,getValRol)
		print("[SENT MESSAGE] (" + current_time + ") " + balasan)
		send_function(balasan)
		try:
			write_log(session,current_time,balasan,0)
		except Exception as e:
			print("[ERROR] LOG SAVING : FAILED")

	if getValNim is False:
		temp = register_nim(session)
		if temp is False:
			auto_commands("SYSTEM GET USER NIM INFORMATION",session,current_time)
		else:
			auto_commands("SYSTEM GET USER NAME INFORMATION",session,current_time)
	elif getValNam is False:
		temp = register_nama(session)
		if temp is False:
			auto_commands("SYSTEM GET USER NAME INFORMATION",session,current_time)
		else:
			auto_commands("SYSTEM FIRST GREETING",session,current_time)

	time.sleep(1)


def main(): #Program utama - main
	print('[PROGRAM IS ONLINE]')
	while True:
		unreadMsg = driver.find_elements(By.XPATH, xpath_unread_msg)
		#Detecting Unread Message in Panel Side then Chat with them
		if unreadMsg and AutoChat == 1:
			for penerima in unreadMsg:
				penerima.click()
				time.sleep(2)
				url = driver.page_source
				soup = bs(url, "lxml")

				div = soup.find_all("div", { "class" : "_22Msk" })[-1]
				full_text = div.find_all("span")
				
				cek = check_xpath(xpath_user_name)

				if(cek):
					nama = driver.find_element(By.XPATH, xpath_user_name).text
				else:
					nama = "Anonymous"

				print(nama)

				try:
					waktu = full_text[-1].find(text=True)
				except IndexError:
					waktu = "-"

				try:
					pesan = full_text[1].find_all(text=True)[-1]
				except IndexError:
					pesan = "You replied last"

				if waktu == "-" or waktu == None:
					pass
				else:
					sub(nama,waktu,pesan,1)
		#Chatting with the Opened Chat Windows
		else:
			try:
				url = driver.page_source
				soup = bs(url, "lxml")
				
				cek = check_xpath(xpath_user_name)

				if(cek):
					nama = driver.find_element(By.XPATH, xpath_user_name).text
				else:
					nama = "Anonymous"

				div = soup.find_all("div", { "class" : "_22Msk" })[-1]
				full_text = div.find_all("span")

				try:
					waktu = full_text[-1].find(text=True)
				except IndexError:
					waktu = "-"

				try:
					pesan = full_text[1].find_all(text=True)[-1]
				except IndexError:
					pesan = "You replied last"

				if waktu == "-" or waktu == None:
					pass
				else:
					sub(nama,waktu,pesan,0)

			except Exception as e:
				pass
		# Else End
	# While End



def messages(): #Pending - Proses mengirim pesan ke suatu individu
	target = input('[Send To] >> ')
	text = input('[Message]    >> ')
	panel = driver.find_element(By.ID, 'pane-side')
	elem = None
	spd = 0
	rep = 0
	while elem is None:
	    if spd < 15000:
	    	spd += 500
	    else:
	    	spd -= 15000
	    	rep += 1
	    	if rep == 3:
	    		break
	    try:
	        driver.execute_script('arguments[0].scrollTop = %s' %spd, panel)
	        elem = driver.find_element(By.XPATH, xpath_title.format(target))
	    except:
	    	pass

	if elem is not None:
		ac = ActionChains(driver)
		ac.move_to_element(elem).click().perform()
		time.sleep(2)
		url = driver.page_source
		user = driver.find_element(By.XPATH, xpath_user_name).text
		driver.find_element(By.XPATH, xpath_msg_box).send_keys(text)
		time.sleep(3)
		driver.find_element(By.XPATH, xpath_send_button).click()



def unread_messages(): #Pending - Proses menghitung jumlah pesan yang belum dibaca
	panel = driver.find_element(By.ID, 'pane-side')
	spd = 0
	count = 0
	prev_one = []
	while spd < 15000:
	    spd += 500
	    unreadMsg = driver.find_elements(By.XPATH, "//*[@class='_2UaNq _2ko65']//*[@data-icon='default-user']")
	    
	    try:
	        driver.execute_script('arguments[0].scrollTop = %s' %spd, panel)
	    except:
	    	pass

	    if unreadMsg:
	    	for msg in unreadMsg:
	    		if len(prev_one) == 0:
	    			prev_one.append(msg)
			    	count += 1
			    	print(len(prev_one))
	    		for checker in prev_one:
			    	if checker != msg:
		    			prev_one.append(msg)
			    		count += 1
			    		print(len(prev_one))

	print('[INFO] Total pesan yang belum dibaca : '+ str(count))
	spd = 0
	try:
	    driver.execute_script('arguments[0].scrollTop = %s' %spd, panel)
	except:
		pass


while True:
	greetings()
	init()
	main()