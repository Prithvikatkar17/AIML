class enplayee:
    start_time = "9 AM"
    end_time = "5 PM"

    def change_time(self,start,end):
        self.start_time = start
        self.end_time = end

class teacher(enplayee):
    def __init__(self,subject):
        self.subject = subject

t1 = teacher("maths")
print(t1.subject,t1.start_time,t1.end_time)
t1.change_time("10 AM","6 PM")
print(t1.subject,t1.start_time,t1.end_time)