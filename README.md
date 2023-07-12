# mTBI (concussion) classification using MRI data

Abstract

Mild Traumatic Brain Injury (mTBI) is a common but often underdiagnosed condition that may lead to significant long-term consequences for the patient if left untreated. In this study, we are developing a machine learning model based on MRI images that accurately predicts mTBI and its associated symptoms based on physician diagnosis. The model is trained and tested on image data from 250 patients who are diagnosed with or without mTBI. This project aims to explore ways to assist physicians in making accurate mTBI diagnoses and identify patterns in MRI images that may help save time in the diagnosis process. These findings from the project will have significant implications for clinical practices as the developed model may provide physicians with a reliable tool for early and accurate mTBI diagnosis, potentially avoiding more expensive diagnostic exams. Additionally, the identified patterns in MRI images may help physicians quickly identify signs of mTBI and its associated symptoms, leading to earlier and more effective treatment. However, initial predictions of mTBI directly from MRI have been poor. One of the reasons is that the majority of the slice images in brain MRI are typically not important for diagnoses, such as slices that include jaw data. We will develop a model that can predict if a slice has clinically important cerebral imaging. We will also explore a two-stage methodology to first predict symptoms; and then from symptoms, predict mTBI. 


Goals

1. Assist physicians in accurately diagnosing mild traumatic brain injury (mTBI) 
2. Save time in preprocessing by automatically identifying patterns in MRI slices
3. Develop a two-stage methodology to first predict symptoms; and then from symptoms, predict mTBI. 
