import os
import subprocess
from pathlib import Path
from langchain_core.tools import tool

from core.config import WORKSPACE_DIR

@tool
def read_file(file_path: str) -> str:
    """读取指定路径的文件内容。当用户要求分析或查看某个文件时，使用此工具读取它。只能读取 workspace 目录下的文件。"""
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = WORKSPACE_DIR / target_path
        target_path = target_path.resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only read files within {WORKSPACE_DIR}"

        if not target_path.exists():
            return f"Error: File {file_path} does not exist."
            
        if not target_path.is_file():
            return f"Error: {file_path} is a directory, not a file."

        if target_path.stat().st_size > 1 * 1024 * 1024:
            return f"Error: File {file_path} is too large (exceeds 1MB). Please request smaller files or split them."

        with open(target_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

@tool
def edit_file(file_path: str, content: str) -> str:
    """写入或覆盖指定路径的文件内容。当用户要求新增或修改某个文件时使用。只能修改 workspace 目录下的文件。"""
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = WORKSPACE_DIR / target_path
        target_path = target_path.resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only edit/create files within {WORKSPACE_DIR}"

        # 确保父目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Success: File {file_path} has been written."
    except Exception as e:
        return f"Error writing file {file_path}: {str(e)}"

@tool
def delete_file(file_path: str) -> str:
    """删除指定路径的文件。当用户明确要求删除某个文件时使用。只能删除 workspace 目录下的文件。"""
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = WORKSPACE_DIR / target_path
        target_path = target_path.resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only delete files within {WORKSPACE_DIR}"

        if not target_path.exists():
            return f"Error: File {file_path} does not exist."

        if target_path.is_dir():
            return f"Error: {file_path} is a directory. This tool can only delete files."

        target_path.unlink()
        return f"Success: File {file_path} has been deleted."
    except Exception as e:
        return f"Error deleting file {file_path}: {str(e)}"

@tool
def execute_shell(command: str) -> str:
    """执行 Shell 脚本或命令。由于安全原因，执行目录将被强制限制在 workspace 下。任何试图通过命令（如 `cd ..` 或绝对路径）修改外部文件的行为都可能受限。"""
    try:
        # 在 workspace 目录下执行命令
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=30  # 限制执行时间防止死循环脚本
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr}"
        
        if not output.strip():
            output = "Command executed successfully with no output."
            
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
def search_files(regex_pattern: str, file_pattern: str = "*", directory: str = ".") -> str:
    """
    使用正则表达式在指定目录及匹配特定文件名模式的文件中搜索内容。
    参数：
    - regex_pattern: 要搜索的正则表达式。
    - file_pattern: 文件名匹配模式，例如 '*.py', '*.ts'。默认为 '*' (所有文件)。
    - directory: 相对 workspace 的子目录路径。默认为 '.' (根目录)。
    返回匹配的内容及行号。
    """
    import fnmatch
    import re
    
    try:
        base_dir = WORKSPACE_DIR / directory
        base_dir = base_dir.resolve()
        
        if not str(base_dir).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only search within {WORKSPACE_DIR}"
            
        if not base_dir.exists() or not base_dir.is_dir():
            return f"Error: Directory {directory} does not exist."

        compiled_regex = re.compile(regex_pattern)
        results = []
        
        # 限制最多搜索 500 个文件或返回 1000 行结果，防止结果过大
        files_scanned = 0
        matches_found = 0
        
        for root, _, files in os.walk(base_dir):
            for filename in files:
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue
                    
                files_scanned += 1
                if files_scanned > 500:
                    results.append("... (Search aborted: Too many files scanned. Please narrow your search directory or file_pattern.)")
                    break
                    
                filepath = Path(root) / filename
                
                # 跳过太大的文件或二进制文件
                try:
                    if filepath.stat().st_size > 1024 * 1024:
                        continue
                        
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        file_matches = []
                        for i, line in enumerate(lines):
                            if compiled_regex.search(line):
                                file_matches.append(f"{i+1}: {line.strip()}")
                                matches_found += 1
                                
                                if matches_found > 1000:
                                    break
                                    
                        if file_matches:
                            rel_path = filepath.relative_to(WORKSPACE_DIR)
                            results.append(f"--- {rel_path} ---")
                            results.extend(file_matches)
                            results.append("")
                            
                except UnicodeDecodeError:
                    pass # 跳过非文本文件
                    
                if matches_found > 1000:
                    results.append("... (Search aborted: Too many matches found. Please refine your regex_pattern.)")
                    break
            else:
                continue
            break
            
        if not results:
            return f"No matches found for pattern '{regex_pattern}' in '{file_pattern}' files under '{directory}'."
            
        return "\n".join(results)
    except Exception as e:
        return f"Error searching files: {str(e)}"

@tool
def replace_in_file(file_path: str, diff: str) -> str:
    """
    修改已存在文件中的特定内容段落。
    参数：
    - file_path: 相对于 workspace 的文件路径。
    - diff: SEARCH/REPLACE 差异块，格式必须完全遵循以下规范：
      ------- SEARCH
      [你要替换掉的精确原文（包括完整的行、空格、缩进）]
      =======
      [你想替换成的新内容]
      +++++++ REPLACE
    注意：SEARCH 块必须精确匹配文件中的原始内容，不能有任何多余或遗漏。此工具只会替换找到的第一个匹配项。
    """
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = WORKSPACE_DIR / target_path
        target_path = target_path.resolve()

        if not str(target_path).startswith(str(WORKSPACE_DIR)):
            return f"Error: Permission denied. Can only edit files within {WORKSPACE_DIR}"

        if not target_path.exists() or not target_path.is_file():
            return f"Error: File {file_path} does not exist."

        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单的 SEARCH/REPLACE 解析逻辑
        search_marker = "------- SEARCH"
        mid_marker = "======="
        replace_marker = "+++++++ REPLACE"
        
        blocks = diff.split(search_marker)
        
        if len(blocks) < 2:
            return "Error: Invalid diff format. Missing '------- SEARCH' marker."
            
        success_count = 0
        new_content = content
        
        for block in blocks[1:]: # 跳过第一个空串
            if mid_marker not in block or replace_marker not in block:
                continue
                
            parts = block.split(mid_marker)
            search_text = parts[0].strip("\n")
            
            replace_parts = parts[1].split(replace_marker)
            replace_text = replace_parts[0].strip("\n")
            
            if not search_text:
                continue
                
            # 严格检查 SEARCH 块是否存在于文件中
            if search_text not in new_content:
                return f"Error: The SEARCH block could not be found exactly as provided in {file_path}. Please make sure whitespace and indentations match perfectly."
                
            # 只替换第一个匹配项
            new_content = new_content.replace(search_text, replace_text, 1)
            success_count += 1
            
        if success_count == 0:
            return "Error: No valid SEARCH/REPLACE blocks were found and executed."

        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        return f"Success: Replaced {success_count} section(s) in {file_path}."
    except Exception as e:
        return f"Error replacing in file {file_path}: {str(e)}"

# 注册可用的工具
tools = [read_file, edit_file, delete_file, execute_shell, search_files, replace_in_file]
tools_by_name = {t.name: t for t in tools}
