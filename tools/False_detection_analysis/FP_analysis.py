import csv

fdlist = []
toollist = []
fddict = {}
proxdict = {}

fdline = ''
proxiline = ''

with open('normal_fn_test.csv', newline='') as fdfile:
    fdreader = csv.reader(fdfile, delimiter=',')
    for fdrow in fdreader:
        fdlist.append(fdrow)
with open('mimikatz.csv', newline='') as toolfile:
    toolreader = csv.reader(toolfile, delimiter=',')
    for toolrow in toolreader:
        toollist.append(toolrow)

with open('fdanalysis.csv', 'w') as fda:

    # Get fdlist length.
    for fdrow in range(len(fdlist)):
        # Create dict key.
        for toolrow in range(len(toollist)):
            fddict[toolrow] = ''
            proxdict[toolrow] = ''

        # Focus on the each fd dll.
        for fdnum, fd in enumerate(fdlist[fdrow]):
            # Get toollist length.
            for toollistnum in range(len(toollist)):
                # Initialize proximity.
                proximity = 0
                # Iterate tool dll.
                for toolnum, tooldll in enumerate(toollist[toollistnum]):
                    # In case of matching fd dll and tool dll.
                    if fd == tooldll:
                        if abs(fdnum - toolnum) == 0:
                            proximity = 4
                        elif abs(fdnum - toolnum) == 1:
                            proximity = 3
                        elif abs(fdnum - toolnum) == 2:
                            proximity = 2
                        else:
                            proximity = 1
                        fddict[toollistnum] = fddict[toollistnum] + fd + ','
                        proxdict[toollistnum] = proxdict[toollistnum] + str(proximity) + ','
                    # In case of not matching fd dll and tool dll.
                    else:
                        # If total proximity is zero, record fd dll and zero proximity.
                        proximity += 0
                        if toolnum == len(toollist[toollistnum]) - 1:
                            if proximity == 0:
                                fddict[toollistnum] = fddict[toollistnum] + fd + ','
                                proxdict[toollistnum] = proxdict[toollistnum] + str(proximity) + ','

        # Delete last comma.
        for toolrow in range(len(toollist)):
            fddict[toolrow] = fddict[toolrow][:-1]
            proxdict[toolrow] = proxdict[toolrow][:-1]

        print(fddict)
        print(proxdict)
        print('')

        for resultnum in range(len(proxdict)):
            fda.write(fddict[resultnum])
            fda.write('\n')
            fda.write(proxdict[resultnum])
            fda.write('\n')
            fda.write('\n')
