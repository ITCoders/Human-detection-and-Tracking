# Human detection and Tracking Türkçe Çeviri (Turkish Translation)

[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/arpit1997)

## Giriş
_Bu projede insan tespiti, yüz tespiti, yüz tanıma ve bireyi takip etme sorunu üzerinde çalıştık. Projemiz, belirli bir videoda bir insanı ve yüzünü tespit edebilmekte ve tespit edilen yüzlerin Yerel ikili Desen Histogramını (LBPH) saklayabilmektedir. LBPH özellikleri, görüntüleri tanımak ve kategorize etmek için kullanılan bir görüntüden çıkarılan kilit noktalardır. Videoda bir insan tespit edildiğinde, ona bir etiket atayan kişinin izini sürdük. Diğer videolarda onları tanımak için bireylerin depolanmış LBPH özelliklerini kullandık. Çeşitli videoları taradıktan sonra programımız, camera1 tarafından çekilen videoda subject1, camera2 tarafından videoda subject1 olarak etiketlenmiş kişi gibi çıktı verir. Bu şekilde bir kişiyi birden fazla kamera tarafından çekilen videoda tanıyarak takip ettik. Tüm çalışmalarımız [openCV] yardımıyla makine öğrenimi ve görüntü işleme uygulamasına dayanmaktadır (http://opencv.org )._ **Bu kod opencv 3.1.1, python 3.4 ve C ++ üzerine kurulmuştur, opencv'nin diğer sürümleri desteklenmez.**
## Gereksinimler
* **opencv [v3.1.1]**
	* ** Linux'ta kurulum:**
			Ubuntu'da opencv'nin tam kurulumu için [buraya tıklayabilirsiniz](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/).
	* ** Windows'ta kurulum**
			Windows'ta opencv'nin tam kurulumu için [buraya tıklayabilirsiniz](https://putuyuwono.wordpress.com/2015/04/23/building-and-installing-opencv-3-0-on-windows-7-64-bit/)
* **python3**
	* Ubuntu'da python 3.4, aşağıda verilen komutla terminal üzerinden kurulabilir:
		`sudo apt-get install python3`
* ** python kütüphaneleri:**
	İşte tüm python kütüphanelerinin bir listesi
	* Python Image Library (PILLOW)
	* Imutils
	* numpy

* **C++**

## Approach
## Yaklaşma
* Kod aşağıda verilen adımları takip eder:
	1. Önce bir video okur ve her kareyi tek tek işler.
	2. Her kare için bir insanı tespit etmeye çalışır. Bir insan tespit edilirse etrafına bir dikdörtgen çizer.
	3. 2. adımı tamamladıktan sonra insan yüzünü tespit etmeye çalışır.
	4. bir insan yüzü tespit edilirse, onu önceden eğitilmiş bir model dosyasıyla tanımaya çalışır.	
	5. İnsan yüzü tanınırsa, etiketi o insan yüzüne koyar, aksi takdirde bir sonraki kare için tekrar 2. adıma geçer
* Depo aşağıdaki gibi yapılandırılmıştır:
	* `main.py` : Bu, insanları algılayan ve tanıyan ana python dosyasıdır.
	* `main.cpp`: Bu, insanları algılayan ve tanıyan ana C ++ dosyasıdır.
	* `create_face_model.py` : Bu python betiği, `data /` klasöründeki verilen verileri kullanarak model dosyası oluşturmak için kullanılır
	* `model.yaml`: Bu dosya, verilen veriler için eğitilmiş model içerir. Bu eğitimli model, verilen veriler için her yüzün LBPH özelliklerini içerir.
	* '`face_cascades/`: Bu dizin, kodlarımızı test etmek için örnek veriler içerir. Bu veriler, bazı videolardan pratiküler bir kişinin yüz görüntüleri çıkarılarak hazırlanır.
	* `scripts/`: Bu dizin, farklı sorunlar üzerinde çalıştığımız bazı yararlı komut dosyaları içerir.
	* `video/`: Bu dizin, test ederken kullandığımız bazı videoları içerir.

## Kurulum

## Python
Yukarıdaki yükleme paragrafında açıklanan gerekli kitaplıkları yüklemeyi unutmayın.

İlk önce çalıştırmanız gerekir create_face_model.py oluşturmak için / data içindeki görüntüleri kullanan bir dosya.yaml dosyası
* Proje klasöründe çalıştırın
```sh 
python create_face_model.py
```
* Kodun python sürümünü çalıştırmak için tüm giriş videolarını tek bir klasöre koymanız ve ardından bu klasörün yolunu komut satırı argümanı olarak sağlamanız gerekir:
```sh
python3 main.py -v /path/to/input/videos/  
```
Örnek- dizin yapımız için:
```sh
 python3 main.py -v /video 
```

## C++
* Kodun C ++ sürümünü openCV ile derlemek için komut:
```sh
 g++ -ggdb `pkg-config --cflags opencv` -o `basename name_of_file.cpp .cpp` name_of_file.cpp `pkg-config --libs opencv` 
```
Örnek- dizin yapımız için:
```sh
 g++ -ggdb `pkg-config --cflags opencv` -o `basename main.cpp .cpp` main.cpp `pkg-config --libs opencv` 
```  
* Kodun C ++ sürümünü çalıştırmak için tüm giriş videolarını tek bir klasöre koymanız ve ardından o videonun yolunu komut satırı argümanı olarak sağlamanız gerekir:
```sh
./name_of_file /path/to/input/video_file 
```  
Örnek- dizin yapımız için:
```sh
 ./main /video/2.mp4
```
* kendi model dosyanızı oluşturma; Kendi model dosyanızı oluşturmak için aşağıda verilen adımları uygulamanız yeterlidir:
	* her birey için görüntüleri şu şekilde yeniden adlandırın: `subjectx.y.jpg` örneğin kişi 1 için resimler şu şekilde adlandırılmalıdır `subject01.0.jpg` , `subject01.1.jpg` ve benzeri.
	* tüm kişilerin tüm resimlerini tek bir klasöre koyun, örneğin görebilirsiniz `data\` klasör daha sonra aşağıda verilen bu komutu çalıştırın:
		`python3 create_face_model.py -i /path/to/persons_images/` 

## Kodun performansı
* Bu bir bilgisayar vizyonu projesi olduğundan, çok fazla hesaplama gücü gerektirir ve kodun performansı burada bir tür sorundur.
* Kod, performansı analiz etmek için iki farklı makinede test edildi. Giriş 30fps 720p video idi.
	* AMD A4 çift çekirdekli işlemcili bir makinede oldukça kötü olan 4 fps çıktı aldık.
	* Intel i5 dört çekirdekli işlemciye sahip bir makinede saniyede 12 kare çıktı elde ettik.

## Sonuçlar
![alt text](https://raw.githubusercontent.com/ITCoders/Human-detection-and-Tracking/master/results/g.jpg "Logo Title Text 1")
![alt text](https://raw.githubusercontent.com/ITCoders/Human-detection-and-Tracking/master/results/k.jpg "Logo Title Text 1")
![alt text](https://raw.githubusercontent.com/ITCoders/Human-detection-and-Tracking/master/results/k.jpg "Logo Title Text 1")
![alt text](https://raw.githubusercontent.com/ITCoders/Human-detection-and-Tracking/master/results/o.jpg "Logo Title Text 1")

Proje raporunu bulabilirsiniz [here](https://github.com/ITCoders/Human-detection-and-Tracking/raw/master/results/HUMAN%20DETECTION%20ANDaRECOGNITION.pdf)
## Gürültü
* kodun performansını artırın
* kodun doğruluğunu artırın ve yanlış pozitif oranı azaltın.
* yüz tanıma doğruluğunu yüzde 90'ın üzerine çıkarın

## Özel Teşekkürler:
* [Jignesh S. Bhatt](http://www.iiitvadodara.ac.in/faculty/jsb001.html) - Bu projeye rehberlik ettiğiniz için teşekkür ederiz
* [Kamal Awasthi](http://github.com/KamalAwasthi) - Kodu test etmede yardımcı oldu
