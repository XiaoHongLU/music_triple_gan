import sys
import json
from subprocess import call


def main(args):
    file = open(args, 'rb')
    s = file.read()
    j = json.loads(s)
    config_file = open('htk_script.scp', 'wb')
    for column in j['columns']:
        for mp3 in j[column]:
            config_file.write('./' + str(mp3) + '    ' + './' + str(mp3) + '.mfcc\n')

    call(['HCopy', '-A', '-D', '-C', 'analysis.config', '-S', 'htk_script.scp'])
    for column in j['columns']:
        for mp3 in j[column]:
            mp3_mfcc = mp3+'.mfcc'
            f = open(mp3+'.txt', 'wb')
            call(['HList', mp3_mfcc], stdout=f)


if __name__ == '__main__':
    arg = sys.argv[1]
    main(arg)