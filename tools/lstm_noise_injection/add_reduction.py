import csv, random, math

def noise_injection(noise_rate, n):
    noise_rate = str(noise_rate).zfill(2)
    path = 'test.csv'
    new_path = 'add_reduce' + str(noise_rate) + '_test_' + str(n) + '.csv'
    all_dll_list = []
    dll_list = []
    csv_file = open(path, "r", encoding="utf-8", errors="", newline="" )
    csv_file = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    next(csv_file)

    for dll in csv_file:
        all_dll_list = all_dll_list + dll[0].split()

    csv_file = open(path, "r", encoding="utf-8", errors="", newline="" )
    csv_file = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    next(csv_file)

    with open(new_path, mode='w') as f:
        f.write('data,tags\n')

        for dll_line in csv_file:
            output = ''
            dlls = dll_line[0].split()
            dll_list.append(dlls[0])
            dll_len = len(dlls)
            deno = 100 / int(noise_rate)
            dll_len_floor = math.floor(dll_len / deno)

            # choice the noise according to dll length from all_dll_list
            noises = random.sample(all_dll_list, dll_len_floor)

            if not noises:
                for dll in dlls:
                    output = output + ' ' + dll
                output = output + ',' + dll_line[1] + '\n'
                f.write(output)

            else:
                for noise in noises:
                    # create random number within dll_len
                    ran_num = random.randint(0, dll_len - 1)
                    del dlls[ran_num]
                    dlls.insert(ran_num, noise)
                for dll in dlls:
                    output = output + ' ' + dll
                output = output + ',' + dll_line[1] + '\n'
                f.write(output)
        f.close()

noise_list = [5,10,15,20,25,30,35,40]
for noise_rate in noise_list:
    for n in range(100):
        noise_injection(noise_rate, n)