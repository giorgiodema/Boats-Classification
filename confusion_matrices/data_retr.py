
#from pprint import pprint as print

f = open('lenet_conf_mat.txt')
x = f.read()


x = x.replace('\t', ' ')

x = x.split('\n')

x = [xi.split(' ')[:-1] for xi in x]


x = [[float(xii) for xii in xi] for xi in x]


x[22].append(0) 

manz = [0 for i in range(23)]

for i in range(23):
	s = 0
	for j in range(23):
		s += x[j][i]
	manz[i] = '-' if s==0 else str(100*x[i][i] / s)[:4]+'%'

d = {'Raccoltarifiuti': 17, 'Barchino': 2, 'Patanella': 15, 'Lanciafino10mMarrone': 8, 'Lanciafino10mBianca': 7, 'Topa': 20, 'Ambulanza': 1, 'Cacciapesca': 3, 'VaporettoACTV': 21, 'Sandoloaremi': 18, 'Mototopo': 14, 'MotoscafoACTV': 13, 'Lanciamaggioredi10mBianca': 9, 'Polizia': 16, 'Alilaguna': 0, 'Motopontonerettangolare': 12, 'Sanpierota': 19, 'Lanciamaggioredi10mMarrone': 10, 'Caorlina': 4, 'VigilidelFuoco': 22, 'Lanciafino10m': 6, 'Gondola': 5, 'Motobarca': 11}
d = {v:k for k,v in d.items()}

for key in sorted(d.keys()):
	print(manz[key], d[key])

