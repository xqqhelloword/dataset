Resumes-A:(data1, label1)
Resumes-B:(data2, label2)
Resumes-C:(data3, label3)

The format of one resume is txt, and using the "###" to segment each resume block such as "basic information","education background","job experience" .
The total categories is 10, each category is labeled with one-hot format.
Each resume has a category which is formatted as a txt file, for example, the resume file :"0.txt", then it's category is saved in the "label" directory using a txt file named "0.txt".
In the category file, each row represents a label of one resume block,.
The dictionary of one-hot index and label is:

label_number = {'base_info': 0, 'edu_back': 1, 'job_exp': 2, 'self_comment': 3, 'school_exp': 4, 'honour': 5,
           'other': 6, 'sci_exp': 7,'skill': 8,'item_exp':9}
number2label = ['base_info', 'edu_back', 'job_exp', 'self_comment', 'school_exp', 'honour',
           'other', 'sci_exp','skill','item_exp']
