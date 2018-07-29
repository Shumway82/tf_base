import io
import json


class ParamsSerializer(object):
    """This class serves as a container base class for serializing attributes with json and
    enables saving and loading a human readable text file.
    """
    def __init__(self, verbose=False):
        self.__description__ = self.__class__.__name__
        self.verbose = verbose

    def save(self, path=None):
        """Function for storing all attributes of this container class into a .json file that
        is a human readable text-file.

            # Arguments
                path : as string what path+file shall be saved (without file-suffix)
        """
        if path is None:
            path = './' + self.__class__.__name__

        try:
            with io.open(path + '.json', 'w', encoding='utf8') as outfile:
                str_ = json.dumps(self.__dict__, indent=4, sort_keys=True, separators=(',', ': '),
                                  ensure_ascii=False)
                outfile.writelines(str(str_))
            self.ps_print(' [*] Pass loading ' + path)
        except BaseException as ex:
            self.ps_print('Saving the object was not possible : ', ex)
            return False

        return True

    def load(self, path=None):
        """Function for loading all attributes of this container class from a .json
        file that is just a human readable text-file saved as a dictionary.

            # Arguments
                path : as string what path+file shall be saved
                    (without file-suffix but have to be .json anyway)
        """
        if path is None:
            path = './' + self.__class__.__name__

        try:
            with open(path + '.json', "r") as data_file:
                data = json.load(data_file)
                self.__dict__ = data
            self.ps_print(' [*] Pass loading ' + path)
            return self
        except BaseException as ex:
            self.ps_print('Loading the .json of parameters was not possible : ', ex)
            return None

    def ps_print(self, *args, **kwargs):
        #if self.verbose:
            print(*args, **kwargs)
