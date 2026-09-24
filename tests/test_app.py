from app import app


def test_login_page_redirects_and_authenticates():
    client = app.test_client()

    resp = client.get('/')
    assert resp.status_code in (200, 302)

    login_resp = client.get('/login')
    assert login_resp.status_code == 200
    assert b'username' in login_resp.data.lower()

    auth_resp = client.post(
        '/login',
        data={'username': 'admin', 'password': 'admin123'},
        follow_redirects=True,
    )
    assert auth_resp.status_code == 200
    assert b'SolarSense Dashboard' in auth_resp.data


def test_history_endpoint_returns_list_for_logged_user():
    client = app.test_client()
    client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)

    resp = client.get('/history')
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)
