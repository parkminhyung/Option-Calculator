from datetime import datetime, timedelta


def calculate_days_to_expiry(expiry_date):
    """만기일까지 남은 일수 계산 (만기일 포함)"""
    if not expiry_date:
        return None
        
    expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
    # 만기일에 하루를 더함
    expiry = expiry + timedelta(days=1)
    today = datetime.now()
    return (expiry - today).days