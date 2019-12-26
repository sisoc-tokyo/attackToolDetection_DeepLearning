import csv, random, math

def noise_injection(noise_rate, n):
    noise_rate = str(noise_rate).zfill(2)
    path = 'test.csv'
    new_path = 'reduction' + str(noise_rate) + '_test_' + str(n) + '.csv'
    csv_file = open(path, "r", encoding="utf-8", errors="", newline="" )
    csv_file = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    next(csv_file)

    n = 0
    with open(new_path, mode='w') as f:
        f.write('data,tags\n')

        for dll_line in csv_file:
            output = ''
            dlls = dll_line[0].split()
            dll_len = len(dlls)
            deno = 100 / int(noise_rate)
            dll_len_floor = math.floor(dll_len / deno)

            for reduction_num in range(dll_len_floor):
                reduction_int = random.randint(0, dll_len - 1)
                del dlls[reduction_int]
                dll_len = dll_len - 1
            for dll in dlls:
                output = output + ' ' + dll
            output = output + ',' + dll_line[1] + '\n'
            f.write(output)
            n += 1
        f.close()

noise_list = [5,10,15,20,25,30,35,40]
for noise_rate in noise_list:
    for n in range(100):
        noise_injection(noise_rate, n)