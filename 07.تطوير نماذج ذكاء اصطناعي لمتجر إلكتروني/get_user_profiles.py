import pandas as pd

def make_connection_with_db():
  
  import mysql.connector
  # الاتصال مع قاعدة البيانات
  connection_mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="wp-ecommerce"
  )
  # مؤشر 
  cursor = connection_mydb.cursor(dictionary=True)
  return connection_mydb,cursor

# دالة جلب مواصفات المستخدمين
def get_users_profiles():
    #إطار بيانات للنتائج
    df=pd.DataFrame(columns=['user_id','country','age','gender'])
    # اتصال ومؤشر على قاعدة البيانات
    _,cursor=make_connection_with_db()
    # الاستعلام عن كافة المستخدمين
    sql="SELECT ID FROM wp_users"
    # تنفيذ الاستعلام
    cursor.execute(sql)
    # جلب كل النتائج
    users_results = cursor.fetchall()
    # الدورن على المستخدمين
    for user in users_results:
            # معرف المستخدم
            user_id=user['ID']
            # استعلام البلد
            sql = "SELECT meta_value FROM wp_usermeta  WHERE user_id=(%s) and meta_key='country'"
            # معامل الاستعلام
            param = (user_id, )
            # تنفيذ الاستعلام
            cursor.execute(sql, param)
            # جلب كل النتائج
            result = cursor.fetchall()
            # اختبار وجود نتيجة
            if result!=None and len(result)>0:
                country=result[0]['meta_value']
            else:
                 country="Unknown"
            # استعلام العمر
            sql = "SELECT meta_value FROM wp_usermeta  WHERE user_id=(%s) and meta_key='age'"
            param = (user_id, )
            cursor.execute(sql, param)
            result = cursor.fetchall()
            if result!=None and len(result)>0:
                age=result[0]['meta_value']
            else:
                 age="Unknown"
            # استعلام الجنس
            sql = "SELECT meta_value FROM wp_usermeta  WHERE user_id=(%s) and meta_key='gender'"
            param = (user_id, )
            cursor.execute(sql, param)
            result = cursor.fetchall()
            if result!=None and len(result)>0:
                gender=result[0]['meta_value']
            else:
                 gender="Unknown"
            # إنشاء غرض مفتاح/قيمة
            obj={
                "user_id":[user['ID']],
                "country":[country],
                "age":[age],
                "gender":[gender],
                }
            df_obj=pd.DataFrame(obj)
            # إضافة صف إلى إطار البيانات
            df=pd.concat([df,df_obj], ignore_index=True)    
    # تنظيف البيانات
    df.drop(df[df['country']=='Unknown'].index)
    df.drop(df[df['age']=='Unknown'].index)
    df.drop(df[df['gender']=='Unknown'].index)
    # تحويل عمود العمر إلى عمود رقمي
    df['age']=pd.to_numeric(df['age'])
    return df
