from tools import *

def write_centroids(directory, a, rW):
	'''Writes all centroid positions for each frame in a .res file
	with the same name as the original video file, appending "_c".
	It requires numpy library (as np) and the get_all_cent function, defined in tools.py file.

	Parameters:
	directory (string): Path to save results
	a (dic): Parsed video using parsing_video function, defined in tools.py file
	rW (float): Radius used for the windowing method'''

	# rW: Windowing radius
    c = get_all_cent(a, rW, prints=True)
    new_name = a['name'][:-4]+'_c.res'
    header_text = "X Centroid 1\tY Centroid 1\tX Centroid 2\tY Centroid 2"
    np.savetxt(directory+new_name, c.transpose(), header=header_text, delimiter='\t')

# Opening and reading configuration file

with open('config.txt', 'r') as file:
    config = file.read()
config = config.split('\n')
content = []
for line in config:
	b = 0
	for l in range(len(line)):
		if line[l] == '=':
			a = l+2
		if line[l] == '#' or line[l] == '\t':
			b = l
			break
	if b == 0:
		c = line[a:]
	else:
		c = line[a:b]

	try:
		content.append(float(c))
	except ValueError:
		content.append(c)
big_directory, folder, overwrite, resol, w_length, D, B, rW, rM = content
params = w_length, D, B
overwrite = True if overwrite == 'True' else False
folder = str(int(folder))
b = B/D
directory = big_directory+folder+'\\'

# Computing and saving results for all files within the directory selected

header_text = '\tobstime\talt_corr\tflong (")\tftran (")\tfwhm (")'
T1 = time.time()
results = []
files = ser_files(directory)
for n in range(1,len(files)):
	a = parsing_video(directory+'\\'+files[n-1])
	obstime = f'{files[n-1][-27:-14]}:{files[n-1][-13:-11]}:{files[n-1][-10:-8]}'

	if a != False:
		file_name = a['name']

		if overwrite:
			write_centroids(directory, a, rW)
		else:
			try:
			    c = np.loadtxt(directory+a['name'][:-4]+'_c.res', delimiter='\t', skiprows=1).transpose()
			except FileNotFoundError:
			    write_centroids(directory, a, rW)

		c = np.loadtxt(directory+a['name'][:-4]+'_c.res', delimiter='\t', skiprows=1).transpose()

		sig_l, sig_t = sigma(a, c, resol)
		sol = fwhm(a, sig_l, sig_t, params)

		print(n, obstime, sol[0], sol[1], sol[2], sol[3])
		results.append([n, obstime, sol[0], sol[1], sol[2], sol[3]])
	else:
		print(n, obstime, alt_corr, np.nan, np.nan, np.nan)
		results.append([n, obstime, alt_corr, np.nan, np.nan, np.nan])

	np.savetxt(big_directory+'\\'+folder+'.res', np.array(results), 
			   fmt='%s', 
			   header=header_text, delimiter='\t')

TIME = (time.time()-T1)/3600
print(f'\nSaved as {folder}.res\t{TIME} h')