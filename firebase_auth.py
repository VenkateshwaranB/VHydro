import streamlit as st
import re
import time
from datetime import datetime

# Hardcoded users for demonstration (replace with proper authentication later)
DEMO_USERS = {
    "user@example.com": "Password123",
    "admin@vhydro.com": "Admin123"
}

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

# Function to handle signup
def signup(email, password, confirm_password):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if password != confirm_password:
        return False, "Passwords don't match"
    
    if not is_strong_password(password):
        return False, "Password must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 digit"
    
    if email in DEMO_USERS:
        return False, "Email already in use"
    
    # Add user to demo users (in a real app, you would save to a database)
    DEMO_USERS[email] = password
    
    # Store user info in session state
    st.session_state["user_id"] = email
    st.session_state["email"] = email
    st.session_state["logged_in"] = True
    
    return True, "Account created successfully!"

# Function to handle login
def login(email, password):
    if not email or not password:
        return False, "Please enter both email and password"
    
    # Check if user exists and password matches
    if email in DEMO_USERS and DEMO_USERS[email] == password:
        # Store user info in session state
        st.session_state["user_id"] = email
        st.session_state["email"] = email
        st.session_state["logged_in"] = True
        
        return True, "Login successful!"
    else:
        return False, "Invalid email or password"

# Function to handle password reset (simplified mock version)
def reset_password(email):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if email in DEMO_USERS:
        return True, "Password reset email would be sent! (Demo only)"
    else:
        return False, "Email not found"

# Function to log out
def logout():
    for key in ["user_id", "email", "logged_in"]:
        if key in st.session_state:
            del st.session_state[key]

# Main authentication function
def authenticate():
    """
    Main authentication function.
    Returns True if user is authenticated, False otherwise.
    """
    # Check if user is already authenticated
    if st.session_state.get("logged_in", False):
        return True
    
    # Otherwise, show auth page
    return auth_page()

# Function to render login/signup UI
def auth_page():
    # CSS for auth forms
    st.markdown("""
    <style>
    .auth-form {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecfb 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #0e4194;
    }
    .auth-header {
        background: linear-gradient(90deg, #0e4194, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
        font-size: 1.8rem;
    }
    
    .auth-input {
        background: white;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 10px 15px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    
    .auth-input:focus {
        border-color: #0e4194;
        box-shadow: 0 0 0 2px rgba(14, 65, 148, 0.2);
    }
    
    .auth-button {
        background: linear-gradient(90deg, #0e4194, #3a7bd5);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 12px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }
    
    .auth-button:hover {
        background: linear-gradient(90deg, #0c3880, #3373c8);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .auth-toggle {
        text-align: center;
        margin-top: 15px;
        color: #666;
    }
    
    .auth-toggle a {
        color: #0e4194;
        text-decoration: none;
        font-weight: 500;
    }
    
    .auth-toggle a:hover {
        text-decoration: underline;
    }
    
    .auth-logo {
        text-align: center;
        margin-bottom: 20px;
    }
    
    .auth-logo img {
        max-width: 120px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add animated logo and title
    st.markdown("""
    <div class="auth-logo">
        <div style="font-size: 4rem; margin-bottom: 10px;">🧪</div>
        <h1 style="color: #0e4194; margin: 0; font-size: 2.2rem;">VHydro</h1>
        <p style="color: #666; margin-top: 5px;">Hydrocarbon Quality Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create centered container for authentication
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Initialize login/signup mode
        if "auth_mode" not in st.session_state:
            st.session_state["auth_mode"] = "login"
            
        # Toggle function
        def toggle_auth_mode():
            st.session_state["auth_mode"] = "signup" if st.session_state["auth_mode"] == "login" else "login"
            
        # Login form
        if st.session_state["auth_mode"] == "login":
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            st.markdown('<h2 class="auth-header">Welcome Back</h2>', unsafe_allow_html=True)
            
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            login_button = st.button("Login", key="login_button", use_container_width=True)
            
            if login_button:
                success, message = login(email, password)
                if success:
                    st.success(message)
                    time.sleep(1)  # Wait for a second to show success message
                    st.rerun()
                else:
                    st.error(message)
                    
            # Forgot password
            forgot_password_expand = st.expander("Forgot Password?")
            with forgot_password_expand:
                reset_email = st.text_input("Enter your email to reset password", key="reset_email")
                if st.button("Send Reset Link", key="reset_button"):
                    if reset_email:
                        success, message = reset_password(reset_email)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter your email")
            
            st.markdown("""
            <div class="auth-toggle">
                Don't have an account? <a href="#" onclick="
                    document.getElementById('switch_to_signup').click(); 
                    return false;">Sign up</a>
            </div>
            """, unsafe_allow_html=True)
            
            # Hidden button for toggling
            if st.button("Switch to Signup", key="switch_to_signup", help="Switch to signup form"):
                toggle_auth_mode()
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Signup form
        else:
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
            
            signup_button = st.button("Sign Up", key="signup_button", use_container_width=True)
            
            if signup_button:
                success, message = signup(new_email, new_password, confirm_password)
                if success:
                    st.success(message)
                    time.sleep(1)  # Wait for a second to show success message
                    st.rerun()
                else:
                    st.error(message)
            
            st.markdown("""
            <div class="auth-toggle">
                Already have an account? <a href="#" onclick="
                    document.getElementById('switch_to_login').click(); 
                    return false;">Log in</a>
            </div>
            """, unsafe_allow_html=True)
            
            # Hidden button for toggling
            if st.button("Switch to Login", key="switch_to_login", help="Switch to login form"):
                toggle_auth_mode()
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add demo credentials for easy testing
    st.markdown("""
    <div style="text-align: center; margin-top: 15px; background: rgba(0, 0, 0, 0.05); 
                padding: 10px; border-radius: 10px; max-width: 400px; margin-left: auto; margin-right: auto;">
        <p style="margin-bottom: 5px; font-weight: bold; color: #666;">Demo Credentials:</p>
        <code>Email: user@example.com<br>Password: Password123</code>
    </div>
    """, unsafe_allow_html=True)
    
    return False

# Function to show user profile/account page
def user_account_page():
    st.markdown("<h2 class='sub-header'>My Account</h2>", unsafe_allow_html=True)
    
    if not st.session_state.get("logged_in", False):
        st.warning("Your session has expired. Please log in again.")
        logout()
        st.rerun()
        return
    
    # Create a card-like interface for user info
    st.markdown("""
    <style>
    .user-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecfb 100%);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #0e4194;
    }
    .user-header {
        border-bottom: 1px solid #e9ecef;
        padding-bottom: 15px;
        margin-bottom: 20px;
    }
    .user-field {
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    .field-label {
        font-weight: bold;
        color: #0e4194;
        width: 120px;
        flex-shrink: 0;
    }
    .field-value {
        flex-grow: 1;
        padding: 10px 15px;
        background: white;
        border-radius: 6px;
        border: 1px solid #e9ecef;
    }
    .user-avatar {
        width: 80px;
        height: 80px;
        background: #3a7bd5;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto 15px auto;
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.markdown('<div class="user-header">', unsafe_allow_html=True)
    
    # Get first initial for avatar
    email = st.session_state.get('email', 'User')
    initial = email[0].upper() if email and len(email) > 0 else 'U'
    
    st.markdown(f"""
    <div class="user-avatar">{initial}</div>
    <h3 style="text-align: center; margin: 0;">{email}</h3>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Email:</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="field-value">{email}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Role:</span>', unsafe_allow_html=True)
    st.markdown('<span class="field-value">Analyst</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Last Login:</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="field-value">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Password change section
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top: 0;">Change Password</h3>', unsafe_allow_html=True)
    
    current_password = st.text_input("Current Password", type="password", key="current_pw")
    new_password = st.text_input("New Password", type="password", key="new_pw")
    confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pw")
    
    # Password requirements notice
    st.markdown("""
    <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 15px;">
    Password must be at least 8 characters with 1 uppercase letter, 1 lowercase letter, and 1 digit.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Update Password", use_container_width=True):
        # This is a demo, so just show a success message
        st.success("Password updated successfully! (Demo only)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Logout section
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'>Logout</h3>", unsafe_allow_html=True)
    st.write("Click the button below to log out of your account.")
    
    if st.button("Logout", key="logout_button", use_container_width=True):
        logout()
        st.success("You have been logged out!")
        time.sleep(1)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
