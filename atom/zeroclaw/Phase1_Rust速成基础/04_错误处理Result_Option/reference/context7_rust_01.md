---
type: context7_documentation
library: Comprehensive Rust (Google)
version: latest
fetched_at: 2026-03-10
knowledge_point: 04_错误处理Result_Option
context7_query: "Result Option error handling ? operator unwrap expect pattern matching"
---

# Context7 文档：Rust 错误处理基础

## 文档来源
- 库名称：Comprehensive Rust (Google)
- Context7 ID: /google/comprehensive-rust

## 关键信息提取

### 1. Result<T, E> 基础
Result 是一个枚举，表示成功（Ok）或失败（Err）。

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_file(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// 模式匹配处理
match read_file("example.txt") {
    Ok(contents) => println!("{contents}"),
    Err(e) => println!("Error: {e}"),
}

// 组合子
let result: Result<i32, &str> = Ok(5);
let doubled = result.map(|x| x * 2);  // Ok(10)

// unwrap 变体
let value = Ok::<i32, &str>(42).unwrap_or(0);

// Option → Result 转换
let opt = Some(5);
let res: Result<i32, &str> = opt.ok_or("No value");
```

### 2. ? 运算符
替代冗长的 match 语句，自动传播错误。如果是 Err，提前返回；如果是 Ok，解包值。

```rust
fn read_username(path: &str) -> Result<String, io::Error> {
    let mut username_file = fs::File::open(path)?;  // Err 时自动返回
    let mut username = String::new();
    username_file.read_to_string(&mut username)?;
    Ok(username)
}
```

### 3. 自定义错误类型 + From trait
通过实现 From trait，? 运算符可以自动转换错误类型。

```rust
#[derive(Debug)]
enum ReadUsernameError {
    IoError(io::Error),
    EmptyUsername(String),
}

impl From<io::Error> for ReadUsernameError {
    fn from(err: io::Error) -> Self {
        Self::IoError(err)
    }
}

fn read_username(path: &str) -> Result<String, ReadUsernameError> {
    let mut username = String::with_capacity(100);
    fs::File::open(path)?.read_to_string(&mut username)?;  // io::Error 自动转换
    if username.is_empty() {
        return Err(ReadUsernameError::EmptyUsername(String::from(path)));
    }
    Ok(username)
}
```

### 4. 表达式求值器示例 - Result 实战
展示 Result 在递归数据结构中的使用。

```rust
#[derive(Debug, PartialEq)]
pub enum Error { DivideByZero }

pub enum Expr {
    Num(i64),
    Add(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn eval(&self) -> Result<i64, Error> {
        match self {
            Expr::Num(n) => Ok(*n),
            Expr::Add(l, r) => Ok(l.eval()? + r.eval()?),
            Expr::Div(l, r) => {
                let rhs = r.eval()?;
                if rhs == 0 { Err(Error::DivideByZero) } else { Ok(l.eval()? / rhs) }
            }
        }
    }
}
```
