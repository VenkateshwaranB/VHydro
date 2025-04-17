import streamlit as st
import pyrebase
import re
import secrets
import time
from datetime import datetime, timedelta

# Firebase configuration - you'll need to replace this with your own Firebase project credentials
# firebaseConfig = {
#     "apiKey": "YOUR_API_KEY",
#     "authDomain": "YOUR_PROJECT_ID.firebaseapp.com",
#     "projectId": "YOUR_PROJECT_ID",
#     "storageBucket": "YOUR_PROJECT_ID.appspot.com",
#     "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",
#     "appId": "YOUR_APP_ID",
#     "databaseURL": "https://YOUR_PROJECT_ID-default-rtdb.firebaseio.com/"  # Required for database interaction
# }

firebaseConfig = {
    "apiKey": "AIzaSyDZnM6nJDP6_aEIfDuHE9Nt-Q7fIxshsI4",
    "authDomain": "vhydro-852ae.firebaseapp.com",
    "projectId": "vhydro-852ae",
    "storageBucket": "vhydro-852ae.firebasestorage.app",
    "messagingSenderId": "539968125295",
    "appId": "1:539968125295:web:af5c5ee6c7ac5e906b509a",
    "measurementId": "G-YTPEB08T3T"
  };

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

# Function to validate email format
def is_valid_email(email):
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_pattern, email))

# Function to validate password strength
def is_strong_password(password):
    # At least 8 characters, with 1 uppercase, 1 lowercase, 1 digit
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

# Function to generate a session token
def generate_session_token():
    return secrets.token_hex(16)

# Function to save session info to Firebase
def save_session(user_id, token, expiry):
    session_data = {
        "token": token,
        "expiry": expiry.isoformat(),
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }
    db.child("sessions").child(token).set(session_data)
    return token

# Function to verify session
def verify_session():
    token = st.session_state.get("token")
    if not token:
        return False
    
    session = db.child("sessions").child(token).get().val()
    if not session:
        return False
    
    expiry = datetime.fromisoformat(session["expiry"])
    if expiry < datetime.now():
        # Expired session
        return False
    
    return session["user_id"]

# Function to log out
def logout():
    token = st.session_state.get("token")
    if token:
        # Remove session from Firebase
        db.child("sessions").child(token).remove()
    
    # Clear session state
    for key in ["token", "user_id", "email"]:
        if key in st.session_state:
            del st.session_state[key]

# Function to handle signup
def signup(email, password, confirm_password):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if password != confirm_password:
        return False, "Passwords don't match"
    
    if not is_strong_password(password):
        return False, "Password must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 digit"
    
    try:
        # Create user in Firebase
        user = auth.create_user_with_email_and_password(email, password)
        
        # Store additional user info in database (optional)
        user_data = {
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat()
        }
        db.child("users").child(user["localId"]).set(user_data)
        
        # Create session
        token = generate_session_token()
        expiry = datetime.now() + timedelta(days=7)  # 1 week session
        save_session(user["localId"], token, expiry)
        
        # Update session state
        st.session_state["token"] = token
        st.session_state["user_id"] = user["localId"]
        st.session_state["email"] = email
        
        return True, "Account created successfully!"
    except Exception as e:
        if "EMAIL_EXISTS" in str(e):
            return False, "Email already in use"
        return False, f"An error occurred: {str(e)}"

# Function to handle login
def login(email, password):
    if not email or not password:
        return False, "Please enter both email and password"
    
    try:
        # Sign in with Firebase
        user = auth.sign_in_with_email_and_password(email, password)
        
        # Create session
        token = generate_session_token()
        expiry = datetime.now() + timedelta(days=7)  # 1 week session
        save_session(user["localId"], token, expiry)
        
        # Update session state
        st.session_state["token"] = token
        st.session_state["user_id"] = user["localId"]
        st.session_state["email"] = email
        
        # Update last login time
        db.child("users").child(user["localId"]).update({"last_login": datetime.now().isoformat()})
        
        return True, "Login successful!"
    except Exception as e:
        if "INVALID_PASSWORD" in str(e) or "EMAIL_NOT_FOUND" in str(e):
            return False, "Invalid email or password"
        return False, f"An error occurred: {str(e)}"

# Function to handle password reset
def reset_password(email):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    try:
        auth.send_password_reset_email(email)
        return True, "Password reset email sent!"
    except Exception as e:
        if "EMAIL_NOT_FOUND" in str(e):
            return False, "Email not found"
        return False, f"An error occurred: {str(e)}"

# Function to render login/signup UI
def auth_page():
    # Check if the user is already authenticated
    if "token" in st.session_state:
        user_id = verify_session()
        if user_id:
            return True
    
    # CSS for auth forms
    st.markdown("""
    <style>
    .auth-form {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .auth-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .form-group {
        margin-bottom: 15px;
    }
    .form-control {
        width: 100%;
        padding: 10px;
        border: 1px solid #ced4da;
        border-radius: 4px;
    }
    .btn-primary {
        background-color: #0e4194;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        cursor: pointer;
    }
    .toggle-link {
        text-align: center;
        margin-top: 15px;
    }
    .reset-link {
        text-align: center;
        margin-top: 10px;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add the VHydro logo at the top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("src/VHydro_Logo.png", width=200)
        except:
            # Fallback if logo is not found
            pass
        st.markdown("<h1 style='text-align: center; color: #0e4194;'>VHydro Login</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Hydrocarbon Quality Prediction</p>", unsafe_allow_html=True)
    
    # Create tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    # Login tab
    with tab1:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        st.markdown('<h2 class="auth-header">Login</h2>', unsafe_allow_html=True)
        
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            login_button = st.button("Login", use_container_width=True)
        
        forgot_password = st.button("Forgot Password?", key="forgot_password")
        
        if login_button:
            success, message = login(email, password)
            if success:
                st.success(message)
                time.sleep(1)  # Wait for a second to show success message
                st.experimental_rerun()
            else:
                st.error(message)
                
        if forgot_password:
            reset_email = st.text_input("Enter your email to reset password", key="reset_email")
            if st.button("Send Reset Link"):
                if reset_email:
                    success, message = reset_password(reset_email)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter your email")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sign Up tab
    with tab2:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        st.markdown('<h2 class="auth-header">Create Account</h2>', unsafe_allow_html=True)
        
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        # Password requirements notice
        st.markdown("""
        <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 15px;">
        Password must be at least 8 characters with 1 uppercase letter, 1 lowercase letter, and 1 digit.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            signup_button = st.button("Sign Up", use_container_width=True)
        
        if signup_button:
            success, message = signup(new_email, new_password, confirm_password)
            if success:
                st.success(message)
                time.sleep(1)  # Wait for a second to show success message
                st.experimental_rerun()
            else:
                st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False

# Function to show user profile/account page
def user_account_page():
    st.markdown("<h2 class='sub-header'>My Account</h2>", unsafe_allow_html=True)
    
    user_id = verify_session()
    if not user_id:
        st.warning("Your session has expired. Please log in again.")
        logout()
        st.experimental_rerun()
        return
    
    # Get user data
    user_data = db.child("users").child(user_id).get().val()
    
    if not user_data:
        st.error("Could not retrieve user data.")
        return
    
    # Create a card-like interface for user info
    st.markdown("""
    <style>
    .user-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-header {
        border-bottom: 1px solid #e9ecef;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .user-field {
        margin-bottom: 15px;
    }
    .field-label {
        font-weight: bold;
        color: #0e4194;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.markdown('<div class="user-header">', unsafe_allow_html=True)
    st.markdown(f"<h3>👤 {user_data.get('email', st.session_state.get('email', 'User'))}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Format timestamps
    created_at = user_data.get('created_at', 'Unknown')
    last_login = user_data.get('last_login', 'Unknown')
    
    try:
        if created_at != 'Unknown':
            created_dt = datetime.fromisoformat(created_at)
            created_at = created_dt.strftime("%B %d, %Y at %I:%M %p")
        
        if last_login != 'Unknown':
            last_dt = datetime.fromisoformat(last_login)
            last_login = last_dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        pass
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Email:</span>', unsafe_allow_html=True)
    st.write(user_data.get('email', st.session_state.get('email', 'Unknown')))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Account Created:</span>', unsafe_allow_html=True)
    st.write(created_at)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Last Login:</span>', unsafe_allow_html=True)
    st.write(last_login)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Password change section
    st.markdown("<h3 class='section-header'>Change Password</h3>", unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.write("To change your password, click the button below to receive a password reset email.")
    
    if st.button("Send Password Reset Email"):
        email = user_data.get('email', st.session_state.get('email'))
        if email:
            success, message = reset_password(email)
            if success:
                st.success(message)
            else:
                st.error(message)
        else:
            st.error("Could not determine your email address.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Logout section
    st.markdown("<h3 class='section-header'>Logout</h3>", unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.write("Click the button below to log out of your account.")
    
    if st.button("Logout"):
        logout()
        st.success("You have been logged out!")
        time.sleep(1)
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main authentication function to use in your app
def authenticate():
    """
    Main authentication function.
    Returns True if user is authenticated, False otherwise.
    """
    # Check if user is already authenticated
    if "token" in st.session_state:
        user_id = verify_session()
        if user_id:
            return True
    
    # Otherwise, show auth page
    return auth_page()
