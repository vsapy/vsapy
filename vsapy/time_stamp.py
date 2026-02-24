import datetime

class TimeStamp(object):
    @staticmethod
    def get_creation_data_time_stamp():
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.datetime.now())

    @staticmethod
    def compare_time_stamps(t1, t2):
        t1obj = datetime.datetime.strptime(t1, 'Timestamp: %Y-%m-%d %H:%M:%S.%f')
        t2obj = datetime.datetime.strptime(t2, 'Timestamp: %Y-%m-%d %H:%M:%S.%f')
        return t1obj.time() == t2obj.time()
