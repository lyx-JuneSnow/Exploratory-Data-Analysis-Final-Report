**文件说明：**

diabetic_data.csv：数据集

IDS_mapping.csv：数据集中分类变量（admission_type_id, discharge_disposition_id, admission_source_id）编码的含义



**数据集说明：**

l **来源**：美国130家医院及综合医疗服务网络（1999-2008年）

l **数据内容**：糖尿病患者的住院记录，涵盖实验室检查、药物治疗、住院时长等信息

l **目标变量**：预测患者是否会在出院后30天内再次入院（二分类任务：readmitted）

l **样本量**：约10万条住院记录

 

**变量说明：**

1. encounter_id ：就诊记录的唯一标识符。 
2. patient_nbr：患者的唯一标识符。 
3. race：患者的种族。 
4. gender：患者的性别。 
5. age：患者的年龄范围（以区间表示）。 
6. weight：患者的体重（单位未知）。 
7. admission_type_id：入院类型的编码，含义见IDS_mapping。 
8. discharge_disposition_id：出院去向的编码，含义见IDS_mapping。 
9. admission_source_id: 入院来源的编码，含义见IDS_mapping。 
10. time_in_hospital：住院天数。 
11. payer_code：支付方的编码（如医保类型）。 
12. medical_specialty：接诊医生的专业领域。 
13. num_lab_procedures：住院期间进行的实验室检查次数。 
14. num_procedures：住院期间进行的非实验室手术或操作次数。 
15. num_medications：住院期间使用的药物数量。 
16. number_outpatient：过去一年内门诊就诊次数。 
17. number_emergency：过去一年内急诊就诊次数。 
18. number_inpatient：过去一年内住院次数。 
19. diag_1：主要诊断编码（ICD-9标准）。 
20. diag_2：次要诊断编码（ICD-9标准）。 
21. diag_3：第三诊断编码（ICD-9标准）。 
22. number_diagnoses：住院期间记录的诊断总数。 
23. max_glu_serum：住院期间最高血糖水平。 
24. A1Cresult：糖化血红蛋白（HbA1c）检测结果。 
25. metformin：是否服用二甲双胍（降糖药）。 

26-39.其他糖尿病相关药物的使用情况（与 `metformin` 类似）。 

40. change：住院期间是否更改过药物。 
41. diabetesMed：是否服用糖尿病药物。 
42. readmitted：患者是否在30天内再次住院。 