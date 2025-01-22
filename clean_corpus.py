import os  
import tempfile  
import shutil  

def clean_corpus_large_file(file_path):  
    """  
    处理大型语料库文件，使用临时文件方式删除单独成行的"图片"文本  
    
    Args:  
        file_path (str): 语料库文件的路径  
        
    Returns:  
        None: 直接修改原文件  
    """  
    # 创建临时文件  
    temp_file = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)  
    
    try:  
        # 逐行读取并处理  
        with open(file_path, 'r', encoding='utf-8') as file:  
            for line in file:  
                # 如果不是单独的"图片"行，则写入临时文件  
                if line.strip() != "图片":  
                    temp_file.write(line)  
        
        temp_file.close()  
        
        # 用临时文件替换原文件  
        shutil.move(temp_file.name, file_path)  
        
    except Exception as e:  
        # 发生错误时删除临时文件  
        os.unlink(temp_file.name)  
        raise e  

# 使用示例  
if __name__ == "__main__":  
    corpus_path = "corpus.txt"  # 替换为实际的文件路径  
    clean_corpus_large_file(corpus_path)  
    print("语料库清理完成！")