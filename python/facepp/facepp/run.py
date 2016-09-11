from utils.prettyprint import print_result
from apps.demo import Demo

def main():

    demo = Demo()
    #faces = demo.create_images('test')
    #demo.train('test')
    #result = demo.recognize('test')
    #print_result('recognition', result)
    #demo.callback('test', faces.iterkeys())
    #demo.delete_group('test')
    print demo.delete_person(['Jim Parsons', 'Leonardo DiCaprio', 'Andy Liu'])
    #demo.callback('test', ['Jim Parsons', 'Leonardo DiCaprio', 'Andy Liu'])

if __name__ == '__main__':
    main()
