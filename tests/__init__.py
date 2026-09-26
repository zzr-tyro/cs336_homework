import pytest

if __name__ == "__main__":
    # 告诉 pytest 去自动运行这个测试函数
    pytest.main(["-k", "test_train_bpe_special_tokens", "-s", "-v"])